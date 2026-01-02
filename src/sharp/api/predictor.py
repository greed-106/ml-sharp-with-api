"""API predictor module - wraps CLI prediction logic for API use.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pillow_heif
import torch
from PIL import Image

from sharp.cli.predict import predict_image
from sharp.utils.io import convert_focallength, extract_exif
from sharp.utils.gaussians import save_ply

LOGGER = logging.getLogger(__name__)


def load_image_from_bytes(
    image_bytes: bytes, auto_rotate: bool = True, remove_alpha: bool = True
) -> tuple[np.ndarray, float]:
    """
    从字节数据加载图片，并提取焦距信息
    
    与 sharp.utils.io.load_rgb 保持完全一致的逻辑
    
    Args:
        image_bytes: 图片字节数据
        auto_rotate: 是否根据 EXIF 自动旋转
        remove_alpha: 是否移除 alpha 通道
    
    Returns:
        (image_array, focal_length_px): RGB 图片数组和焦距（像素单位）
    """
    # 检测文件类型并加载图片
    bytes_io = io.BytesIO(image_bytes)
    
    # 尝试检测是否为 HEIC 格式
    try:
        # 重置指针
        bytes_io.seek(0)
        # 尝试作为 HEIC 打开
        heif_file = pillow_heif.open_heif(bytes_io, convert_hdr_to_8bit=True)
        img_pil = heif_file.to_pillow()
        LOGGER.info("Loaded HEIC image")
    except Exception:
        # 不是 HEIC，使用 PIL 打开
        bytes_io.seek(0)
        img_pil = Image.open(bytes_io)
    
    # 提取 EXIF 信息
    img_exif = extract_exif(img_pil)
    
    # 根据 EXIF 旋转图片
    if auto_rotate:
        exif_orientation = img_exif.get("Orientation", 1)
        if exif_orientation == 3:
            img_pil = img_pil.transpose(Image.ROTATE_180)
        elif exif_orientation == 6:
            img_pil = img_pil.transpose(Image.ROTATE_270)
        elif exif_orientation == 8:
            img_pil = img_pil.transpose(Image.ROTATE_90)
        elif exif_orientation != 1:
            LOGGER.warning(f"Ignoring image orientation {exif_orientation}.")
    
    # 提取焦距信息（与 sharp.utils.io.load_rgb 完全一致）
    f_35mm = img_exif.get("FocalLengthIn35mmFilm", img_exif.get("FocalLenIn35mmFilm", None))
    if f_35mm is None or f_35mm < 1:
        f_35mm = img_exif.get("FocalLength", None)
        if f_35mm is None:
            LOGGER.warning("Did not find focal length in EXIF data - Setting to 30mm.")
            f_35mm = 30.0
        if f_35mm < 10.0:
            LOGGER.info("Found focal length below 10mm, assuming it's not for 35mm.")
            # This is a very crude approximation.
            f_35mm *= 8.4
    
    # 转换为 numpy 数组
    img = np.asarray(img_pil)
    
    # 转换为 RGB（如果是单通道）
    if img.ndim < 3 or img.shape[2] == 1:
        img = np.dstack((img, img, img))
    
    # 移除 alpha 通道
    if remove_alpha:
        img = img[:, :, :3]
    
    # 转换焦距为像素单位
    height, width = img.shape[:2]
    f_px = convert_focallength(width, height, f_35mm)
    
    LOGGER.info(f"Loaded image: {width}x{height}, focal_length: {f_35mm}mm @ 35mm film -> {f_px:.2f}px")
    
    return img, f_px


def predict_from_image_bytes(
    predictor,
    image_bytes: bytes,
    device: torch.device,
    output_path: Path,
    task_id: str,
) -> Path:
    """
    从图片字节数据预测并保存 PLY 文件
    
    与 CLI 的处理流程保持一致
    
    Args:
        predictor: 预测器模型
        image_bytes: 图片字节数据
        device: 计算设备
        output_path: 输出目录
        task_id: 任务 ID（用于文件命名）
    
    Returns:
        保存的 PLY 文件路径
    """
    # 加载图片并提取焦距（与 CLI 一致）
    image, f_px = load_image_from_bytes(image_bytes)
    height, width = image.shape[:2]
    
    LOGGER.info(f"Processing image: {width}x{height}, focal_length: {f_px:.2f}px")
    
    # 执行预测（使用 CLI 的 predict_image 函数）
    gaussians = predict_image(predictor, image, f_px, device)
    
    # 保存 PLY 文件
    output_path.mkdir(exist_ok=True, parents=True)
    ply_path = output_path / f"{task_id}.ply"
    
    save_ply(gaussians, f_px, (height, width), ply_path)
    
    LOGGER.info(f"Saved PLY to {ply_path}")
    
    return ply_path
