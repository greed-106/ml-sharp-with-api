# SHARP API 服务

基于 FastAPI 的 SHARP 3D Gaussian Splatting 预测服务，支持实时状态推送和自动压缩。

## 功能特性

- ✅ 异步任务队列处理
- ✅ 图片缓存在内存中
- ✅ 模型懒加载（首次请求时加载）
- ✅ 支持多个任务排队处理
- ✅ **SSE 实时状态推送**（无需轮询）
- ✅ **自动压缩为 .sog 格式**
- ✅ **多 worker 并发压缩**
- ✅ **配置文件管理**
- ✅ RESTful API 接口
- ✅ 任务状态查询
- ✅ 结果文件下载

## 安装依赖

```bash
# 安装 Python 依赖
pip install fastapi uvicorn python-multipart pillow sse-starlette

# 安装 splat-transform（用于压缩）
npm install splat-transform


## 配置文件

首次使用前，复制配置模板并根据需要修改：

```bash
# 复制 JSONC 模板（推荐）
cp config.template.jsonc config.jsonc
```

**注意：** `config.json` 文件支持 JSONC 格式（JSON with Comments），可以直接包含注释。配置加载时会自动移除注释后解析。

支持的注释格式：
- 单行注释：`// 这是注释`
- 多行注释：`/* 这是多行注释 */`

**示例配置文件：**
```jsonc
{
  // 服务器配置
  "server": {
    "host": "0.0.0.0",  // 监听所有网络接口
    "port": 8000
  },
  
  /* 模型配置
     可以指定本地模型路径 */
  "model": {
    "checkpoint_path": "sharp_2572gikvuh.pt",
    "device": "auto"  // 自动选择设备
  }
}
```

### 配置说明

配置文件支持 JSONC 格式（带注释的 JSON）：

```jsonc
{
  // 服务器配置
  "server": {
    "host": "0.0.0.0",           // 服务器监听地址
    "port": 8000,                 // 服务器端口
    "log_level": "INFO"           // 日志级别
  },
  
  // 模型配置
  "model": {
    "checkpoint_path": "sharp_2572gikvuh.pt",  // 模型路径，如果文件不存在则自动下载
    "device": "auto"              // 设备: auto, cuda, mps, cpu
  },
  
  // 存储配置
  "storage": {
    "output_dir": "output/results",  // 输出目录
    "keep_ply": false,            // 压缩后是否保留 PLY 文件
    "db_path": "output/sqlite/metadata.db"  // 元数据数据库路径
  },
  
  // 压缩配置
  "compression": {
    "enabled": true,              // 是否启用压缩
    "command": "npx",             // 压缩命令
    "args_template": [            // 命令参数模板
      "splat-transform",
      "{input_file}",
      "{output_file}"
    ],
    "max_workers": 2,             // 最大并发压缩任务数
    "timeout": 300                // 压缩超时时间（秒）
  }
}
```

详细配置说明请参考 `config.template.jsonc` 文件中的注释。

## 启动服务

### 方式 1: 使用启动脚本

```bash
python start_api.py

# 或指定端口
python start_api.py 8080
```

### 方式 2: 使用 uvicorn

```bash
uvicorn sharp.api.server:app --host 0.0.0.0 --port 8000 --reload
```

服务将在 `http://localhost:8000` 启动。

## API 端点

### 1. 提交预测任务

**POST** `/predict`

上传图片并创建预测任务。

**参数:**
- `file`: 图片文件（multipart/form-data）
- `focal_length`: 可选，焦距（像素单位）

**响应:**
```json
{
  "task_id": "uuid",
  "status": "queued",
  "message": "null"
}
```

### 2. 实时状态推送（SSE）

**GET** `/stream/{task_id}`

通过 Server-Sent Events 实时推送任务状态，无需轮询。

**智能任务查找：**
- 如果任务在队列中，实时推送状态更新
- 如果任务不在队列中，检查结果文件是否存在：
  - 文件存在：返回 `completed` 状态（任务已完成）
  - 文件不存在：返回 404（任务不存在）

**事件格式:**
```json
// 正常状态（queued, converting, compressing, completed）
{
  "status": "converting"
}

// 失败状态（包含错误信息）
{
  "status": "failed",
  "message": "Compression failed: ..."
}
```

**状态值:**
- `queued`: 等待处理
- `converting`: 正在转换为 3D Gaussian
- `compressing`: 正在压缩为 .sog 格式
- `completed`: 处理完成
- `failed`: 处理失败


### 3. 下载结果

**GET** `/result/{task_id}`

下载处理完成的文件（.sog 或 .ply）。


### 4. 获取元数据

**GET** `/metadata/{task_id}`

获取PLY文件的元数据信息（内参矩阵和外参矩阵）。

**响应:**
```json
{
  "task_id": "abc123-def456-ghi789",
  "intrinsic_matrix": [1234.56, 0.0, 960.0, 0.0, 1234.56, 540.0, 0.0, 0.0, 1.0],
  "extrinsic_matrix": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  "created_at": "2025-01-08 12:34:56"
}
```

**说明:**
- `task_id`: 任务ID
- `intrinsic_matrix`: 3x3相机内参矩阵（按行展开为9个元素的数组）
- `extrinsic_matrix`: 4x4相机外参矩阵（按行展开为16个元素的数组，默认为单位矩阵）
- `created_at`: 创建时间戳


### 5. 健康检查

**GET** `/health`

检查服务健康状态。

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "compression_enabled": true,
  "compression_workers": 2
}
```

## 使用示例

### Python 客户端（SSE 实时推送）

```python
import json
import requests
from sseclient import SSEClient

# 1. 提交任务
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    task_info = response.json()
    task_id = task_info["task_id"]

# 2. 通过 SSE 接收实时状态
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

# 3. 下载结果
if data["status"] == "completed":
    response = requests.get(f"http://localhost:8000/result/{task_id}")
    with open("output.sog", "wb") as f:
        f.write(response.content)
```

### JavaScript 客户端（SSE）

```javascript
// 1. 提交任务
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});
const taskInfo = await response.json();

// 2. 监听 SSE 状态更新
const eventSource = new EventSource(`http://localhost:8000/stream/${taskInfo.task_id}`);

eventSource.addEventListener('status', (event) => {
  const data = JSON.parse(event.data);
  
  if (data.status === 'failed') {
    console.error('Task failed:', data.message);
    eventSource.close();
  } else if (data.status === 'completed') {
    console.log('Task completed!');
    eventSource.close();
    // 下载结果
    window.location.href = `http://localhost:8000/result/${taskInfo.task_id}`;
  } else {
    console.log('Status:', data.status);
  }
});
```

### cURL

```bash
# 提交任务
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "focal_length=1000"

# SSE 监听（需要支持 SSE 的工具）
curl -N "http://localhost:8000/stream/{task_id}"

# 查询状态（轮询方式）
curl "http://localhost:8000/status/{task_id}"

# 下载结果
curl "http://localhost:8000/result/{task_id}" -o output.sog
```

## API 文档

启动服务后，访问以下地址查看自动生成的 API 文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 架构说明

### 处理流程

1. **接收请求** → 图片缓存到内存，任务加入队列
2. **3D 转换** → 使用 SHARP 模型转换为 3D Gaussian
3. **保存 PLY** → 异步保存 PLY 文件
4. **压缩** → 使用 splat-transform 压缩为 .sog 格式
5. **清理** → 删除原始 PLY 文件（可配置）

### 模型加载

- 采用懒加载策略，首次请求时才加载模型
- 模型加载后常驻内存，避免重复加载
- 使用异步锁确保模型只加载一次

### 任务队列

- 使用 `asyncio.Queue` 实现异步任务队列
- 后台任务处理器顺序处理转换任务
- 独立的压缩队列支持多 worker 并发压缩
- 图片数据缓存在内存中

### SSE 状态推送

- 每个任务维护独立的状态队列
- 实时推送状态变化，无需客户端轮询
- 任务完成或失败后自动断开 SSE 连接
- 任务数据在 SSE 断开后自动清理（延迟 2 秒确保客户端收到最后状态）

### 并发处理

- 转换任务顺序处理（避免 GPU 资源竞争）
- 压缩任务并发处理（可配置 worker 数量）
- 适合中低并发场景

## 生产环境部署建议

1. **使用 Gunicorn + Uvicorn workers**
   ```bash
   gunicorn sharp.api.server:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **配置 Nginx 反向代理**
   ```nginx
   location / {
       proxy_pass http://localhost:8000;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_set_header Host $host;
       proxy_read_timeout 600s;  # SSE 长连接
   }
   ```

3. **添加认证和限流**
   - 使用 API Key 认证
   - 实现请求限流

4. **监控和日志**
   - 集成日志系统
   - 添加性能监控

5. **高并发优化**
   - 迁移到 Redis 队列
   - 使用对象存储（S3/OSS）
   - 部署多个 worker 实例

## 注意事项

- 模型文件较大（约 2GB），首次启动会自动下载
- GPU 推理速度更快，建议使用 CUDA 设备
- 压缩需要安装 `splat-transform`（npm 包）
- 内存缓存适合低并发场景，高并发请迁移到 Redis
- 结果文件保存在配置的 `output_dir` 目录
- SSE 连接需要保持打开状态直到任务完成
