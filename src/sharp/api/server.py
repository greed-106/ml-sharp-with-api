"""FastAPI server for SHARP 3D Gaussian Splatting prediction.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections import deque
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import save_ply

from .config import get_config
from .predictor import load_image_from_bytes, predict_from_image_bytes

LOGGER = logging.getLogger(__name__)

# 全局模型实例（懒加载）
_model_instance = None
_model_lock = asyncio.Lock()
_device = None

# 任务队列和结果
task_queue = asyncio.Queue()
task_results = {}  # task_id -> result
task_status_queues = {}  # task_id -> asyncio.Queue for SSE

# 压缩任务队列
compression_queue = asyncio.Queue()
compression_semaphore = None  # 将在启动时初始化


class TaskStatus(BaseModel):
    """任务状态模型"""
    task_id: str
    status: str  # queued, converting, compressing, completed, failed
    message: Optional[str] = None


async def broadcast_status(task_id: str, status: str, message: str = ""):
    """
    广播任务状态更新
    
    Args:
        task_id: 任务 ID
        status: 状态值 (queued, converting, compressing, completed, failed)
        message: 消息（仅在 failed 状态时需要，用于传递错误信息）
    """
    # 更新任务状态（用于 SSE 初始状态查询）
    if task_id in task_results:
        task_results[task_id]["status"] = status
        if status == "failed":
            task_results[task_id]["message"] = message
    
    # 发送到 SSE 队列
    if task_id in task_status_queues:
        data = {"status": status}
        if status == "failed" and message:
            data["message"] = message
        await task_status_queues[task_id].put(data)


async def cleanup_task(task_id: str):
    """清理已完成的任务数据"""
    # 等待一小段时间，确保 SSE 客户端收到最后的状态
    await asyncio.sleep(2)
    
    # 删除任务状态
    task_results.pop(task_id, None)
    task_status_queues.pop(task_id, None)
    
    LOGGER.info(f"Cleaned up task {task_id}")


async def get_model():
    """懒加载模型实例"""
    global _model_instance, _device
    
    config = get_config()
    
    async with _model_lock:
        if _model_instance is None:
            LOGGER.info("Loading model for the first time...")
            
            # 确定设备
            device_config = config.model.device.lower()
            if device_config == "auto":
                if torch.cuda.is_available():
                    _device = torch.device("cuda")
                elif torch.mps.is_available():
                    _device = torch.device("mps")
                else:
                    _device = torch.device("cpu")
            else:
                _device = torch.device(device_config)
            
            LOGGER.info(f"Using device: {_device}")
            
            # 加载模型
            checkpoint_path = config.model.checkpoint_path
            
            if checkpoint_path:
                # 检查文件是否存在
                checkpoint_file = Path(checkpoint_path)
                if checkpoint_file.exists():
                    LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
                    state_dict = torch.load(checkpoint_path, weights_only=True)
                else:
                    LOGGER.warning(f"Checkpoint file {checkpoint_path} not found, downloading default model...")
                    LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
                    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
            else:
                LOGGER.info(f"No checkpoint path specified, downloading model from {DEFAULT_MODEL_URL}")
                state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
            
            # 创建预测器
            _model_instance = create_predictor(PredictorParams())
            _model_instance.load_state_dict(state_dict)
            _model_instance.eval()
            _model_instance.to(_device)
            
            LOGGER.info("Model loaded successfully")
    
    return _model_instance, _device


async def compress_file(input_path: Path, output_path: Path, task_id: str):
    """异步压缩文件"""
    config = get_config()
    
    if not config.compression.enabled:
        LOGGER.info(f"Compression disabled, skipping for task {task_id}")
        return input_path
    
    try:
        await broadcast_status(task_id, "compressing")
        
        # 构建命令参数
        args = [
            arg.format(input_file=str(input_path), output_file=str(output_path))
            for arg in config.compression.args_template
        ]
        
        LOGGER.info(f"Compressing {input_path} -> {output_path}")
        LOGGER.info(f"Command: {config.compression.command} {' '.join(args)}")
        
        # 执行压缩命令
        process = await asyncio.create_subprocess_exec(
            config.compression.command,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.compression.timeout
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"Compression failed: {error_msg}")
            
            LOGGER.info(f"Compression completed for task {task_id}")
            
            # 删除原始 PLY 文件（如果配置要求）
            if not config.storage.keep_ply and input_path.exists():
                input_path.unlink()
                LOGGER.info(f"Deleted original PLY file: {input_path}")
            
            return output_path
            
        except asyncio.TimeoutError:
            process.kill()
            raise Exception(f"Compression timeout after {config.compression.timeout}s")
            
    except Exception as e:
        LOGGER.error(f"Compression error for task {task_id}: {str(e)}")
        raise


async def process_compression_queue():
    """后台压缩任务处理器"""
    config = get_config()
    
    while True:
        task = await compression_queue.get()
        
        async with compression_semaphore:
            task_id = task["task_id"]
            ply_path = task["ply_path"]
            
            try:
                # 生成压缩文件路径
                output_path = ply_path.with_suffix(".sog")
                
                # 执行压缩
                await compress_file(ply_path, output_path, task_id)
                
                await broadcast_status(task_id, "completed")
                
                # 异步清理任务（延迟清理，确保 SSE 客户端收到完成状态）
                asyncio.create_task(cleanup_task(task_id))
                
            except Exception as e:
                error_msg = str(e)
                LOGGER.error(f"Compression failed for task {task_id}: {error_msg}")
                await broadcast_status(task_id, "failed", error_msg)
                
                # 失败的任务也需要清理
                asyncio.create_task(cleanup_task(task_id))
            
            finally:
                compression_queue.task_done()


async def process_task_queue():
    """后台任务处理器"""
    config = get_config()
    
    while True:
        task = await task_queue.get()
        task_id = task["task_id"]
        image_bytes = task["image_bytes"]
        
        try:
            LOGGER.info(f"Processing task {task_id}")
            await broadcast_status(task_id, "converting")
            
            # 获取模型
            model, device = await get_model()
            
            # 准备输出路径
            output_dir = Path(config.storage.output_dir)
            
            # 执行预测（使用与 CLI 一致的逻辑）
            ply_path = await asyncio.to_thread(
                predict_from_image_bytes,
                model,
                image_bytes,
                device,
                output_dir,
                task_id
            )
            
            LOGGER.info(f"PLY saved for task {task_id}: {ply_path}")
            
            # 如果启用压缩，添加到压缩队列
            if config.compression.enabled:
                await compression_queue.put({
                    "task_id": task_id,
                    "ply_path": ply_path
                })
                LOGGER.info(f"Task {task_id} added to compression queue")
            else:
                # 不压缩，直接完成
                await broadcast_status(task_id, "completed")
                
                # 异步清理任务
                asyncio.create_task(cleanup_task(task_id))
            
        except Exception as e:
            error_msg = str(e)
            LOGGER.error(f"Task {task_id} failed: {error_msg}", exc_info=True)
            await broadcast_status(task_id, "failed", error_msg)
            
            # 失败的任务也需要清理
            asyncio.create_task(cleanup_task(task_id))
        
        finally:
            task_queue.task_done()


# 创建 FastAPI 应用
app = FastAPI(
    title="SHARP 3D Gaussian Splatting API",
    description="API for converting single images to 3D Gaussian Splatting scenes",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """启动时创建后台任务处理器"""
    global compression_semaphore
    
    config = get_config()
    
    # 初始化压缩信号量
    compression_semaphore = asyncio.Semaphore(config.compression.max_workers)
    
    # 启动任务处理器
    asyncio.create_task(process_task_queue())
    asyncio.create_task(process_compression_queue())
    
    LOGGER.info("Task queue processors started")
    LOGGER.info(f"Compression workers: {config.compression.max_workers}")
    LOGGER.info(f"Output directory: {config.storage.output_dir}")


@app.post("/predict", response_model=TaskStatus)
async def predict(file: UploadFile = File(...)):
    """
    提交图片进行 3D Gaussian Splatting 预测
    
    - **file**: 输入图片文件
    
    图片的焦距信息将自动从 EXIF 提取，与 CLI 工具保持一致
    
    返回任务 ID，可通过 SSE 接收实时更新
    """
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # 读取图片到内存
        image_bytes = await file.read()
        
        # 创建任务
        task_id = str(uuid4())
        task_results[task_id] = {
            "task_id": task_id,
            "status": "queued"
        }
        
        # 添加到队列
        await task_queue.put({
            "task_id": task_id,
            "image_bytes": image_bytes
        })
        
        LOGGER.info(f"Task {task_id} queued. Queue size: {task_queue.qsize()}")
        
        return TaskStatus(**task_results[task_id])
        
    except Exception as e:
        LOGGER.error(f"Error processing upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/stream/{task_id}")
async def stream_status(task_id: str):
    """
    通过 SSE 实时推送任务状态
    
    - **task_id**: 任务 ID
    
    连接建立后会实时推送任务状态，任务完成或失败后自动断开连接
    
    如果任务不在队列中，会检查结果文件是否存在：
    - 如果文件存在，说明任务已完成，直接返回 completed 状态
    - 如果文件不存在，说明任务不存在，返回 404
    """
    config = get_config()
    output_dir = Path(config.storage.output_dir)
    
    # 检查任务是否在队列中
    if task_id not in task_results:
        # 任务不在队列中，检查文件是否存在
        sog_path = output_dir / f"{task_id}.sog"
        ply_path = output_dir / f"{task_id}.ply"
        
        if sog_path.exists() or ply_path.exists():
            # 文件存在，说明任务已完成
            LOGGER.info(f"Task {task_id} not in queue but result file exists, returning completed status")
            
            async def completed_event_generator():
                yield {
                    "event": "status",
                    "data": {"status": "completed"}
                }
            
            return EventSourceResponse(completed_event_generator())
        else:
            # 文件不存在，任务确实不存在
            raise HTTPException(status_code=404, detail="Task not found")
    
    # 任务在队列中，正常处理
    # 创建状态队列
    status_queue = asyncio.Queue()
    task_status_queues[task_id] = status_queue
    
    async def event_generator():
        try:
            # 首先发送当前状态
            if task_id in task_results:
                current_status = task_results[task_id]
                data = {"status": current_status["status"]}
                # 只在 failed 状态时包含 message
                if current_status["status"] == "failed" and "message" in current_status:
                    data["message"] = current_status["message"]
                
                yield {
                    "event": "status",
                    "data": data
                }
            
            # 持续推送更新
            while True:
                update = await status_queue.get()
                yield {
                    "event": "status",
                    "data": update
                }
                
                # 如果任务完成或失败，结束流
                if update["status"] in ["completed", "failed"]:
                    LOGGER.info(f"SSE stream ending for task {task_id}, status: {update['status']}")
                    break
                    
        except asyncio.CancelledError:
            LOGGER.info(f"SSE stream cancelled for task {task_id}")
        finally:
            # 清理 SSE 队列引用
            task_status_queues.pop(task_id, None)
            LOGGER.info(f"SSE stream closed for task {task_id}")
    
    return EventSourceResponse(event_generator())


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """
    下载处理结果文件
    
    - **task_id**: 任务 ID
    
    直接从文件系统查找结果文件（.sog 或 .ply），不依赖任务队列
    """
    config = get_config()
    output_dir = Path(config.storage.output_dir)
    
    # 优先查找压缩后的 .sog 文件
    sog_path = output_dir / f"{task_id}.sog"
    if sog_path.exists():
        return FileResponse(
            path=sog_path,
            media_type="application/octet-stream",
            filename=sog_path.name
        )
    
    # 如果没有 .sog，查找 .ply 文件
    ply_path = output_dir / f"{task_id}.ply"
    if ply_path.exists():
        return FileResponse(
            path=ply_path,
            media_type="application/octet-stream",
            filename=ply_path.name
        )
    
    # 文件不存在
    raise HTTPException(status_code=404, detail="Result file not found")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    config = get_config()
    return {
        "status": "healthy",
        "model_loaded": _model_instance is not None,
        "device": str(_device) if _device else "not initialized",
        "compression_enabled": config.compression.enabled,
        "compression_workers": config.compression.max_workers
    }


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    config = get_config()
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port
    )
