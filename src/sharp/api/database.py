"""Database module for storing PLY metadata.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import aiosqlite

LOGGER = logging.getLogger(__name__)


class MetadataDB:
    """SQLite数据库管理类，用于存储PLY文件元数据"""
    
    def __init__(self, db_path: Path):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self):
        """建立数据库连接"""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            LOGGER.info(f"Database connection established: {self.db_path}")
    
    async def close(self):
        """关闭数据库连接"""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            LOGGER.info("Database connection closed")
    
    async def initialize(self):
        """创建数据库表"""
        await self.connect()
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                task_id TEXT PRIMARY KEY,
                intrinsic_matrix TEXT NOT NULL,
                extrinsic_matrix TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await self._connection.commit()
        LOGGER.info(f"Database initialized at {self.db_path}")
    
    async def insert_metadata(
        self,
        task_id: str,
        intrinsic_matrix: list[float],
        extrinsic_matrix: list[float]
    ):
        """
        插入或更新元数据
        
        Args:
            task_id: 任务ID
            intrinsic_matrix: 3x3内参矩阵（9个元素的列表）
            extrinsic_matrix: 4x4外参矩阵（16个元素的列表）
        """
        await self.connect()
        await self._connection.execute("""
            INSERT OR REPLACE INTO metadata 
            (task_id, intrinsic_matrix, extrinsic_matrix)
            VALUES (?, ?, ?)
        """, (
            task_id,
            json.dumps(intrinsic_matrix),
            json.dumps(extrinsic_matrix)
        ))
        await self._connection.commit()
        LOGGER.info(f"Metadata saved for task {task_id}")
    
    async def get_metadata(self, task_id: str) -> Optional[dict]:
        """
        获取元数据
        
        Args:
            task_id: 任务ID
        
        Returns:
            元数据字典，如果不存在返回None
        """
        await self.connect()
        self._connection.row_factory = aiosqlite.Row
        async with self._connection.execute(
            "SELECT * FROM metadata WHERE task_id = ?",
            (task_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    "task_id": row["task_id"],
                    "intrinsic_matrix": json.loads(row["intrinsic_matrix"]),
                    "extrinsic_matrix": json.loads(row["extrinsic_matrix"]),
                    "created_at": row["created_at"]
                }
            return None
    
    async def delete_metadata(self, task_id: str) -> bool:
        """
        删除元数据
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否删除成功
        """
        await self.connect()
        cursor = await self._connection.execute(
            "DELETE FROM metadata WHERE task_id = ?",
            (task_id,)
        )
        await self._connection.commit()
        return cursor.rowcount > 0


# 全局数据库实例
_db_instance: Optional[MetadataDB] = None


def get_db() -> MetadataDB:
    """获取全局数据库实例"""
    global _db_instance
    if _db_instance is None:
        raise RuntimeError("Database not initialized. Call initialize_db() first.")
    return _db_instance


async def initialize_db(db_path: Path):
    """初始化全局数据库实例"""
    global _db_instance
    _db_instance = MetadataDB(db_path)
    await _db_instance.initialize()


async def close_db():
    """关闭全局数据库连接"""
    global _db_instance
    if _db_instance is not None:
        await _db_instance.close()

