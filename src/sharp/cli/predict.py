"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16"], case_sensitive=False),
    default="fp32",
    help="Model precision for inference. fp16 reduces memory usage and may improve speed.",
)
@click.option(
    "--gpu-postprocessing/--cpu-postprocessing",
    "gpu_postprocessing",
    is_flag=True,
    default=True,
    help="Use GPU for postprocessing (faster) or CPU with FP64 (more stable). Default: GPU.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    precision: str,
    gpu_postprocessing: bool,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)
    
    # 应用 FP16 优化
    if precision.lower() == "fp16":
        if device == "cpu":
            LOGGER.warning("FP16 on CPU may be slow. Consider using CUDA/MPS for better performance.")
        LOGGER.info("Converting model to FP16 precision")
        gaussian_predictor = gaussian_predictor.half()

    output_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        
        # 根据精度选项决定是否使用 FP16
        use_fp16 = precision.lower() == "fp16"
        
        # 根据设备类型决定是否使用 GPU 后处理
        use_gpu_post = gpu_postprocessing and device in ["cuda", "mps"]
        if gpu_postprocessing and device == "cpu":
            LOGGER.info("GPU postprocessing disabled on CPU device.")
            use_gpu_post = False
        
        gaussians = predict_image(
            gaussian_predictor,
            image,
            f_px,
            torch.device(device),
            use_fp16=use_fp16,
            use_gpu_postprocessing=use_gpu_post,
        )

        LOGGER.info("Saving 3DGS to %s", output_path)
        # 使用新的命名格式: originname_{f_px}_{image_width}_{image_height}.ply
        output_filename = f"{image_path.stem}_{int(f_px)}_{width}_{height}.ply"
        save_ply(gaussians, f_px, (height, width), output_path / output_filename)

        if with_rendering:
            output_video_path = (output_path / image_path.stem).with_suffix(".mp4")
            LOGGER.info("Rendering trajectory to %s", output_video_path)

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, output_video_path)


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    use_fp16: bool = False,
    use_gpu_postprocessing: bool = True,
) -> Gaussians3D:
    """Predict Gaussians from an image.
    
    Args:
        predictor: 预测器模型
        image: 输入图片 (H, W, 3) numpy 数组
        f_px: 焦距（像素单位）
        device: 计算设备
        use_fp16: 是否使用 FP16 精度进行推理
        use_gpu_postprocessing: 是否在 GPU 上执行后处理（更快但可能有数值误差）
    
    Returns:
        Gaussians3D: 预测的高斯点云
    """
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    
    # 根据精度选择数据类型
    dtype = torch.float16 if use_fp16 else torch.float32
    
    image_pt = torch.from_numpy(image.copy()).to(dtype).to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).to(dtype).to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()  # 后处理始终使用 float32
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc,
        torch.eye(4).to(device),
        intrinsics_resized,
        internal_shape,
        use_gpu=use_gpu_postprocessing,
    )

    return gaussians
