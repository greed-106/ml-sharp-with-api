"""配置管理模块"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


def strip_json_comments(json_str: str) -> str:
    """
    移除 JSON 字符串中的注释
    
    支持：
    - 单行注释: // comment
    - 多行注释: /* comment */
    """
    # 移除多行注释
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    # 移除单行注释
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    return json_str


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = Field(default="0.0.0.0", description="服务器监听地址")
    port: int = Field(default=8000, description="服务器端口")
    log_level: str = Field(default="INFO", description="日志级别")


class ModelConfig(BaseModel):
    """模型配置"""
    checkpoint_path: Optional[str] = Field(default=None, description="模型检查点路径，为空则自动下载")
    device: str = Field(default="auto", description="设备类型: auto, cuda, mps, cpu")
    use_fp16: bool = Field(default=False, description="是否使用 FP16 精度推理（减少显存占用）")
    use_gpu_postprocessing: bool = Field(default=True, description="是否在 GPU 上执行后处理（更快）")


class StorageConfig(BaseModel):
    """存储配置"""
    output_dir: str = Field(default="output/results", description="输出文件目录")
    keep_ply: bool = Field(default=False, description="压缩后是否保留原始 PLY 文件")
    db_path: str = Field(default="output/sqlite/metadata.db", description="元数据数据库路径")


class CompressionConfig(BaseModel):
    """压缩配置"""
    enabled: bool = Field(default=True, description="是否启用压缩")
    command: str = Field(default="npx", description="压缩命令")
    args_template: list[str] = Field(
        default=["splat-transform", "{input_file}", "{output_file}"],
        description="压缩命令参数模板"
    )
    max_workers: int = Field(default=2, description="最大并发压缩任务数")
    timeout: int = Field(default=300, description="压缩超时时间（秒）")


class Config(BaseModel):
    """完整配置"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)


# 全局配置实例
_config: Optional[Config] = None


def load_config(config_path: str | Path = "config.jsonc") -> Config:
    """加载配置文件（支持 JSONC 格式）"""
    global _config
    
    config_path = Path(config_path)
    
    if config_path.exists():
        LOGGER.info(f"Loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_text = f.read()
        
        # 移除注释
        config_text_clean = strip_json_comments(config_text)
        
        # 解析 JSON
        config_data = json.loads(config_text_clean)
        _config = Config(**config_data)
    else:
        LOGGER.warning(f"Config file {config_path} not found, using defaults")
        _config = Config()
    
    return _config


def get_config() -> Config:
    """获取配置实例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def save_config_template(output_path: str | Path = "config.template.json"):
    """保存配置模板"""
    config = Config()
    output_path = Path(output_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            config.model_dump(),
            f,
            indent=2,
            ensure_ascii=False
        )
    
    LOGGER.info(f"Config template saved to {output_path}")


if __name__ == "__main__":
    # 生成配置模板
    save_config_template("config.template.json")
    print("配置模板已生成: config.template.json")
