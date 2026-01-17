"""Contains linear algebra related utility functions.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def rotation_matrices_from_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert batch of quaternions into rotations matrices.

    Args:
        quaternions: The quaternions convert to matrices.

    Returns:
        The rotations matrices corresponding to the (normalized) quaternions.
    """
    device = quaternions.device
    shape = quaternions.shape[:-1]

    quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
    real_part = quaternions[..., 0]
    vector_part = quaternions[..., 1:]

    vector_cross = get_cross_product_matrix(vector_part)
    real_part = real_part[..., None, None]

    matrix_outer = vector_part[..., :, None] * vector_part[..., None, :]
    matrix_diag = real_part.square() * eyes(3, shape=shape, device=device)
    matrix_cross_1 = 2 * real_part * vector_cross
    matrix_cross_2 = vector_cross @ vector_cross

    return matrix_outer + matrix_diag + matrix_cross_1 + matrix_cross_2


def quaternions_from_rotation_matrices(
    matrices: torch.Tensor, use_gpu: bool = True
) -> torch.Tensor:
    """Convert batch of rotation matrices to quaternions.

    Args:
        matrices: The matrices to convert to quaternions.
        use_gpu: If True, use GPU-accelerated PyTorch implementation.
                 If False, use scipy on CPU (more stable but slower).

    Returns:
        The quaternions corresponding to the rotation matrices.

    Note: CPU path is not differentiable and will be performed on the CPU.
    """
    if not matrices.shape[-2:] == (3, 3):
        raise ValueError(f"matrices have invalid shape {matrices.shape}")

    if use_gpu and matrices.device.type in ["cuda", "mps"]:
        # GPU 加速路径：使用 PyTorch 实现
        # 基于 Shepperd's method (最稳定的四元数提取算法)
        # Reference: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        try:
            return _quaternions_from_rotation_matrices_gpu(matrices)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"GPU quaternion conversion failed: {e}. Falling back to CPU."
            )
            # 失败则回退到 CPU 路径

    # CPU 路径：使用 scipy（原始实现）
    matrices_np = matrices.detach().cpu().numpy()
    quaternions_np = Rotation.from_matrix(matrices_np.reshape(-1, 3, 3)).as_quat()
    # We use a convention where the w component is at the start of the quaternion.
    quaternions_np = quaternions_np[:, [3, 0, 1, 2]]
    quaternions_np = quaternions_np.reshape(matrices_np.shape[:-2] + (4,))
    return torch.as_tensor(quaternions_np, device=matrices.device, dtype=matrices.dtype)


def _quaternions_from_rotation_matrices_gpu(matrices: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated quaternion extraction using Shepperd's method.

    This is a numerically stable method that avoids division by small numbers.
    """
    batch_shape = matrices.shape[:-2]
    matrices = matrices.reshape(-1, 3, 3)

    # Shepperd's method: choose the largest diagonal element to avoid division by zero
    # Trace of rotation matrix
    trace = matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]

    # Pre-allocate quaternion tensor (w, x, y, z)
    q = torch.zeros(matrices.shape[0], 4, device=matrices.device, dtype=matrices.dtype)

    # Case 1: trace > 0 (most common case)
    mask0 = trace > 0
    if mask0.any():
        s = torch.sqrt(trace[mask0] + 1.0) * 2  # s = 4 * w
        q[mask0, 0] = 0.25 * s  # w
        q[mask0, 1] = (matrices[mask0, 2, 1] - matrices[mask0, 1, 2]) / s  # x
        q[mask0, 2] = (matrices[mask0, 0, 2] - matrices[mask0, 2, 0]) / s  # y
        q[mask0, 3] = (matrices[mask0, 1, 0] - matrices[mask0, 0, 1]) / s  # z

    # Case 2: m[0,0] is largest diagonal
    mask1 = (~mask0) & (matrices[:, 0, 0] > matrices[:, 1, 1]) & (
        matrices[:, 0, 0] > matrices[:, 2, 2]
    )
    if mask1.any():
        s = torch.sqrt(
            1.0 + matrices[mask1, 0, 0] - matrices[mask1, 1, 1] - matrices[mask1, 2, 2]
        ) * 2  # s = 4 * x
        q[mask1, 0] = (matrices[mask1, 2, 1] - matrices[mask1, 1, 2]) / s  # w
        q[mask1, 1] = 0.25 * s  # x
        q[mask1, 2] = (matrices[mask1, 0, 1] + matrices[mask1, 1, 0]) / s  # y
        q[mask1, 3] = (matrices[mask1, 0, 2] + matrices[mask1, 2, 0]) / s  # z

    # Case 3: m[1,1] is largest diagonal
    mask2 = (~mask0) & (~mask1) & (matrices[:, 1, 1] > matrices[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(
            1.0 + matrices[mask2, 1, 1] - matrices[mask2, 0, 0] - matrices[mask2, 2, 2]
        ) * 2  # s = 4 * y
        q[mask2, 0] = (matrices[mask2, 0, 2] - matrices[mask2, 2, 0]) / s  # w
        q[mask2, 1] = (matrices[mask2, 0, 1] + matrices[mask2, 1, 0]) / s  # x
        q[mask2, 2] = 0.25 * s  # y
        q[mask2, 3] = (matrices[mask2, 1, 2] + matrices[mask2, 2, 1]) / s  # z

    # Case 4: m[2,2] is largest diagonal
    mask3 = (~mask0) & (~mask1) & (~mask2)
    if mask3.any():
        s = torch.sqrt(
            1.0 + matrices[mask3, 2, 2] - matrices[mask3, 0, 0] - matrices[mask3, 1, 1]
        ) * 2  # s = 4 * z
        q[mask3, 0] = (matrices[mask3, 1, 0] - matrices[mask3, 0, 1]) / s  # w
        q[mask3, 1] = (matrices[mask3, 0, 2] + matrices[mask3, 2, 0]) / s  # x
        q[mask3, 2] = (matrices[mask3, 1, 2] + matrices[mask3, 2, 1]) / s  # y
        q[mask3, 3] = 0.25 * s  # z

    # Normalize quaternions
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)

    # Reshape back to original batch shape
    q = q.reshape(batch_shape + (4,))

    return q


def get_cross_product_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Generate cross product matrix for vector exterior product."""
    if not vectors.shape[-1] == 3:
        raise ValueError("Only 3-dimensional vectors are supported")
    device = vectors.device
    shape = vectors.shape[:-1]
    unit_basis = eyes(3, shape=shape, device=device)
    # We compute the matrix by multiplying each column of unit_basis with the
    # corresponding vector.
    return torch.cross(vectors[..., :, None], unit_basis, dim=-2)


def eyes(
    dim: int, shape: tuple[int, ...], device: torch.device | str | None = None
) -> torch.Tensor:
    """Create batch of identity matrices."""
    return torch.eye(dim, device=device).broadcast_to(shape + (dim, dim)).clone()


def quaternion_product(q1, q2):
    """Compute dot product between two quaternions."""
    real_1 = q1[..., :1]
    real_2 = q2[..., :1]
    vector_1 = q1[..., 1:]
    vector_2 = q2[..., 1:]

    real_out = real_1 * real_2 - (vector_1 * vector_2).sum(dim=-1, keepdim=True)
    vector_out = real_1 * vector_2 + real_2 * vector_1 + torch.cross(vector_1, vector_2)
    return torch.concatenate([real_out, vector_out], dim=-1)


def quaternion_conj(q):
    """Get conjugate of a quaternion."""
    real = q[..., :1]
    vector = q[..., 1:]
    return torch.concatenate([real, -vector], dim=-1)


def project(u: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project tensor u to unit basis a."""
    unit_u = F.normalize(u, dim=-1)
    inner_prod = (unit_u * basis).sum(dim=-1, keepdim=True)
    return inner_prod * u
