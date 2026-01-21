"""Contains `sharp render-single` CLI implementation for rendering single images.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click
import numpy as np
import torch

from sharp.utils import gsplat, io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, load_ply

LOGGER = logging.getLogger(__name__)


def parse_intrinsics_from_filename(filename: str) -> tuple[float, int, int] | None:
    """从文件名解析相机内参。
    
    文件名格式: {name}_{f_px}_{width}_{height}.ply
    
    Args:
        filename: PLY文件名
        
    Returns:
        (f_px, width, height) 或 None（如果解析失败）
    """
    # 匹配格式: xxx_{数字}_{数字}_{数字}.ply
    pattern = r"^(.+)_(\d+)_(\d+)_(\d+)\.ply$"
    match = re.match(pattern, filename)
    
    if match:
        _, f_px_str, width_str, height_str = match.groups()
        return float(f_px_str), int(width_str), int(height_str)
    
    return None


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a directory containing PLY files.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the rendered PNG images.",
    required=True,
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run on. ['cpu', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def render_single_cli(input_path: Path, output_path: Path, device: str, verbose: bool):
    """Render single PNG images from PLY files at original viewpoint."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA not available, falling back to CPU.")
        device = "cpu"

    if not input_path.is_dir():
        LOGGER.error("Input path must be a directory containing PLY files.")
        exit(1)

    output_path.mkdir(exist_ok=True, parents=True)

    # 查找所有PLY文件
    ply_files = list(input_path.glob("*.ply"))
    
    if len(ply_files) == 0:
        LOGGER.info("No PLY files found in %s", input_path)
        return

    LOGGER.info("Found %d PLY files to render.", len(ply_files))

    for ply_path in ply_files:
        LOGGER.info("Rendering %s", ply_path.name)
        
        # 从文件名解析内参
        intrinsics_data = parse_intrinsics_from_filename(ply_path.name)
        
        if intrinsics_data is None:
            LOGGER.warning(
                "Cannot parse intrinsics from filename %s. "
                "Expected format: {name}_{f_px}_{width}_{height}.ply. Skipping.",
                ply_path.name,
            )
            continue
        
        f_px, width, height = intrinsics_data
        LOGGER.info("  Intrinsics: f_px=%.1f, resolution=%dx%d", f_px, width, height)
        
        # 加载PLY文件
        try:
            gaussians, _ = load_ply(ply_path)
        except Exception as e:
            LOGGER.error("Failed to load %s: %s. Skipping.", ply_path.name, e)
            continue
        
        # 渲染单张图片
        output_image_path = output_path / f"{ply_path.stem}.png"
        render_single_image(
            gaussians=gaussians,
            f_px=f_px,
            width=width,
            height=height,
            output_path=output_image_path,
            device=device,
        )
        
        LOGGER.info("  Saved to %s", output_image_path)


def render_single_image(
    gaussians: Gaussians3D,
    f_px: float,
    width: int,
    height: int,
    output_path: Path,
    device: str = "cuda",
) -> None:
    """在原始视角渲染单张图片。
    
    Args:
        gaussians: 3D高斯点云
        f_px: 焦距（像素单位）
        width: 图像宽度
        height: 图像高度
        output_path: 输出PNG文件路径
        device: 计算设备
    """
    device_obj = torch.device(device)
    
    # 构建相机内参矩阵
    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2.0, 0],
            [0, f_px, (height - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device_obj,
        dtype=torch.float32,
    )
    
    # 使用单位矩阵作为外参（原始视角）
    extrinsics = torch.eye(4, device=device_obj, dtype=torch.float32)
    
    # 创建渲染器（使用sRGB色彩空间）
    renderer = gsplat.GSplatRenderer(color_space="sRGB")
    
    # 渲染
    rendering_output = renderer(
        gaussians.to(device_obj),
        extrinsics=extrinsics[None],
        intrinsics=intrinsics[None],
        image_width=width,
        image_height=height,
    )
    
    # 转换为numpy数组并保存
    color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
    color_np = color.detach().cpu().numpy()
    
    io.save_image(color_np, output_path)
