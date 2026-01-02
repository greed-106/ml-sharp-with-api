"""SHARP API module.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

__all__ = ["app"]


def get_app():
    """延迟导入 app，避免在导入模块时就加载所有依赖"""
    from .server import app
    return app
