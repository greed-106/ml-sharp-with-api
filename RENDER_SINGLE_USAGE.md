# render-single 命令使用说明

## 功能
从PLY文件渲染单张PNG图片，相机位于原始视角（外参为4x4单位矩阵），内参从文件名中解析。

## 文件名格式要求
PLY文件必须遵循以下命名格式：
```
{原始名称}_{焦距}_{宽度}_{高度}.ply
```

示例：
- `photo_512_1920_1080.ply` - 焦距512px，分辨率1920x1080
- `image_001_768_2048_1536.ply` - 焦距768px，分辨率2048x1536

## 使用方法

### 基本用法
```bash
sharp render-single -i <输入目录> -o <输出目录>
```

### 参数说明
- `-i, --input-path`: 包含PLY文件的输入目录（必需）
- `-o, --output-path`: 保存PNG图片的输出目录（必需）
- `--device`: 计算设备，可选 'cuda' 或 'cpu'（默认：cuda）
- `-v, --verbose`: 启用详细日志

### 示例

1. 渲染目录中的所有PLY文件：
```bash
sharp render-single -i ./output -o ./rendered
```

2. 使用CPU渲染：
```bash
sharp render-single -i ./output -o ./rendered --device cpu
```

3. 启用详细日志：
```bash
sharp render-single -i ./output -o ./rendered -v
```

## 输出
- 每个PLY文件会生成一个对应的PNG图片
- 输出文件名与输入PLY文件名相同（不含内参后缀）
- 例如：`photo_512_1920_1080.ply` → `photo_512_1920_1080.png`

## 渲染设置
- **外参**：4x4单位矩阵（原始视角）
- **内参**：从文件名解析（焦距、图像宽度、高度）
- **色彩空间**：sRGB
- **输出格式**：PNG

## 注意事项
1. 如果文件名格式不符合要求，该文件会被跳过并显示警告
2. CUDA不可用时会自动回退到CPU
3. 输出目录会自动创建（如果不存在）
