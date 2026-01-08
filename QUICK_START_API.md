# SHARP API 快速入门

## 1. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements-api.txt

# 安装压缩工具（可选，如果需要压缩功能）
npm install splat-transform
```

## 2. 配置服务

```bash
# 复制配置模板
cp config.template.jsonc config.jsonc

# 编辑配置文件（可选）
# config.jsonc 支持 JSONC 格式，可以直接包含注释
# 默认配置已经可以直接使用
```

### 主要配置项

- `model.checkpoint_path`: 模型文件路径（默认 `sharp_2572gikvuh.pt`），如果文件不存在则自动下载
- `compression.enabled`: 是否启用压缩（默认 true）
- `compression.max_workers`: 并发压缩任务数（默认 2）
- `storage.keep_ply`: 压缩后是否保留 PLY 文件（默认 false）
- `storage.output_dir`: 输出文件目录（默认 output/results）
- `model.device`: 设备选择（默认 auto，自动选择 cuda/mps/cpu）

### 元数据存储

API会自动将每个生成的PLY文件的元数据（内参矩阵、外参矩阵、图像分辨率等）存储到SQLite数据库中：
- 数据库位置：可在配置文件中设置 `storage.db_path`（默认：`output/sqlite/metadata.db`）
- 只有通过API生成的文件才会记录元数据
- 可通过 `/metadata/{task_id}` 端点查询

## 3. 启动服务

```bash
python start_api.py
```

服务将在 `http://localhost:8000` 启动。

## 4. 测试服务

### 方法 1: 使用示例客户端（推荐）

```bash
# SSE 实时推送模式
python src/sharp/api/client_example.py data/teaser.jpg

# 轮询模式
python src/sharp/api/client_example.py data/teaser.jpg --polling
```

### 方法 2: 使用 cURL

```bash
# 提交任务
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/teaser.jpg"

# 记录返回的 task_id，然后查询状态
curl "http://localhost:8000/status/{task_id}"

# 下载结果
curl "http://localhost:8000/result/{task_id}" -o output.sog
```

### 方法 3: 使用浏览器

访问 `http://localhost:8000/docs` 使用交互式 API 文档。

## 处理流程

```
上传图片 → 排队等待 → 3D 转换 → 保存 PLY → 压缩为 .sog → 完成
   ↓          ↓           ↓          ↓           ↓          ↓
 queued   queued    converting  converting  compressing completed
```

## 6. SSE 实时状态推送

客户端可以通过 SSE 接收实时状态更新，无需轮询：

```python
from sseclient import SSEClient
import json

# 连接 SSE 端点
sse_url = f"http://localhost:8000/stream/{task_id}"
messages = SSEClient(sse_url)

for msg in messages:
    if msg.event == "status":
        data = json.loads(msg.data)
        status = data["status"]
        
        if status == "failed":
            print(f"Error: {data['message']}")
        else:
            print(f"Status: {status}")
        
        if status in ["completed", "failed"]:
            break
```

## 7. 常见问题

### Q: 压缩失败怎么办？

A: 检查是否安装了 `splat-transform`：
```bash
npm install splat-transform
```

或者在 `config.json` 中禁用压缩：
```json
{
  "compression": {
    "enabled": false
  }
}
```

### Q: 如何使用本地模型文件？

A: 在 `config.json` 中设置模型路径（默认已配置）：
```json
{
  "model": {
    "checkpoint_path": "sharp_2572gikvuh.pt"
  }
}
```

如果文件不存在，服务会自动下载。你也可以指定其他路径：
```json
{
  "model": {
    "checkpoint_path": "path/to/your/model.pt"
  }
}
```

### Q: 如何调整并发压缩数量？

A: 修改 `config.json`：
```json
{
  "compression": {
    "max_workers": 4
  }
}
```

### Q: 输出文件在哪里？

A: 默认在 `output/results/` 目录，可在配置文件中修改。

### Q: 如何保留原始 PLY 文件？

A: 修改 `config.json`：
```json
{
  "storage": {
    "keep_ply": true
  }
}
```


## 更多信息

详细文档请参考 `src/sharp/api/README.md`
