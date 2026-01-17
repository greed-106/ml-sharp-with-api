# Postprocessing 优化说明（简化版）

## 问题分析

原始实现中存在**两个**主要性能瓶颈：

### 1. SVD 分解在 CPU 上执行

```python
# 原始代码：强制在 CPU 上用 FP64 计算
covariance_matrices = covariance_matrices.detach().cpu().to(torch.float64)
rotations, singular_values_2, _ = torch.linalg.svd(covariance_matrices)
```

### 2. 四元数转换使用 scipy（必须在 CPU）

```python
# 原始代码：使用 scipy，必须在 CPU 上
matrices_np = matrices.detach().cpu().numpy()
quaternions_np = Rotation.from_matrix(matrices_np).as_quat()
```

## 优化方案

### 1. GPU 加速的 SVD 分解

直接在 GPU 上执行 SVD，避免数据传输：

```python
# 优化后：直接在 GPU 上计算
rotations, singular_values_2, _ = torch.linalg.svd(covariance_matrices)
```

### 2. GPU 加速的四元数转换

实现了纯 PyTorch 的 **Shepperd's method**，完全在 GPU 上执行：

```python
# 优化后：使用 PyTorch 实现，完全在 GPU 上
def _quaternions_from_rotation_matrices_gpu(matrices: torch.Tensor):
    # 基于 trace 和对角元素选择最稳定的计算路径
    trace = matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]
    # ... 4 种情况的分支处理
```

**Shepperd's method** 是数值最稳定的四元数提取算法，通过选择最大的对角元素避免除以接近零的数。

### 3. 内存分配优化

避免重复创建单位矩阵：

```python
# 优化前：每次创建单位矩阵
diagonal_matrix = torch.eye(3, device=device) * singular_values[..., :, None]

# 优化后：直接构建对角矩阵
diagonal_matrix = torch.zeros(batch_shape + (3, 3), device=device)
diagonal_matrix[..., 0, 0] = singular_values[..., 0]
diagonal_matrix[..., 1, 1] = singular_values[..., 1]
diagonal_matrix[..., 2, 2] = singular_values[..., 2]
```

### CLI 选项

新增 `--gpu-postprocessing` / `--cpu-postprocessing` 选项：

```bash
# 使用 GPU 后处理（默认，推荐）
sharp predict -i image.jpg -o output/ --gpu-postprocessing

# 使用 CPU 后处理（更稳定但慢）
sharp predict -i image.jpg -o output/ --cpu-postprocessing
```


### 何时使用 CPU 后处理？

- 遇到数值不稳定问题（极少见）
- 需要最高精度的结果
- GPU 内存不足

## API 兼容性

API 调用默认使用 GPU 后处理，通过配置文件控制：

```jsonc
{
  "model": {
    "use_gpu_postprocessing": true
  }
}
```

## 技术细节

### 完整的后处理流程

```
1. 构建协方差矩阵: Σ = R @ diag(s²) @ R^T
2. 应用变换: Σ' = T @ Σ @ T^T
3. SVD 分解: Σ' = U @ diag(λ) @ U^T  ← 瓶颈 1
4. 提取四元数: R' = matrix_to_quaternion(U)  ← 瓶颈 2
5. 提取尺度: s' = sqrt(λ)
```

### 数值稳定性

GPU FP32 SVD 在绝大多数情况下足够稳定。如果遇到问题：
1. 代码会自动捕获异常
2. 回退到 CPU FP64 路径
3. 记录警告日志
