# Ascend C Standalone Operator: `linalg_eigh` (Symmetric Eigenvalue Decomposition)

本仓库为华为 2026 电子设计竞赛 **`linalg_eigh`（对称矩阵特征值与特征向量分解）** 算子的独立原生 Ascend C 开发工程。

---

## 1. 目标平台与评价指标

- **目标硬件**: 华为昇腾 910B (CANN 9.0, Ascend C 编程模型)
- **数学问题**: 对实对称矩阵 $A = A^T \in \mathbb{R}^{n \times n}$ 进行特征分解：
  $$A = Q \Lambda Q^T, \quad Q^T Q = I$$
  其中 $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$ 且 $\lambda_1 \le \lambda_2 \le \dots \le \lambda_n$。
- **精度标准**:
  - 特征值相对误差: $\max_i \frac{|\hat{\lambda}_i - \lambda_i|}{\max(1.0, |\lambda_i|)} \le 10^{-4}$
  - 正交性误差: $\|\hat{Q}^T \hat{Q} - I\|_F \le 10^{-4}$
  - 重构误差: $\frac{\|A - \hat{Q} \hat{\Lambda} \hat{Q}^T\|_F}{\|A\|_F} \le 10^{-4}$
- **独立工程设计**:
  - 本仓库专为单算子独立开发打造，已剔除其他 8 个算子和外部无关模块干扰。
  - `op_kernel/linalg_eigh.cpp` 提供最基础的单核/多核极简骨架，`Process()` 方法留空，供开发者/AI 从头设计。

---

## 2. 输入与输出张量定义

| 张量名称 | 属性 | 数据类型 | 格式 | 形状说明 |
| :--- | :--- | :--- | :--- | :--- |
| **`A`** | 输入 (必选) | `float32` | ND | `[..., n, n]` (实对称矩阵) |
| **`eigenvalues`** | 输出 (必选) | `float32` | ND | `[..., n]` (升序排列的特征值) |
| **`eigenvectors`**| 输出 (必选) | `float32` | ND | `[..., n, n]` (正交特征向量矩阵 $Q$) |

- **属性**:
  - `UPLO`: 字符串，`"L"`（默认，读取下三角并镜像）或 `"U"`（读取上三角并镜像）。

---

## 3. 目录结构

```text
ascend-symmetric-eigen/
├── README.md                  # 本文档
├── specs/
│   └── eigh_spec.md           # 详细数学规格与边界形状说明
├── docs/
│   └── ASCEND_C_API.md        # 常用 Ascend C 官方底层 API 速查手册
├── op_host/
│   └── linalg_eigh.cpp        # Host 侧 Tiling 切分与 Workspace 内存分配
├── op_kernel/
│   ├── linalg_eigh_tiling.h   # TilingData 结构体定义
│   ├── tiling_key_linalg_eigh.h # TilingKey 宏选择器
│   └── linalg_eigh.cpp        # Device 侧 Ascend C Kernel 实现入口
└── test/
    └── verify_local.py        # Python 本地数值基准对照验证脚本
```

---

## 4. 推荐数值解法思路

1. **小矩阵 ($n \le 64$): 经典/双侧 Jacobi 旋转法**:
   - 迭代通过 Givens 旋转消去非对角元：$A_{k+1} = G_k^T A_k G_k$，$Q_{k+1} = Q_k G_k$；
   - 每次旋转正交性天然守恒，重构精度高，向量化极容易实现。
2. **中大矩阵 ($64 < n \le 512$): Householder 三对角化 + QR / 分治法**:
   - 先通过 $n-2$ 次 Householder 反射将 $A$ 正交相似变换为对称三对角矩阵 $T = V^T A V$；
   - 对三对角矩阵 $T$ 进行隐式对称位移 QR 迭代（Implicitly Shifted QR）或分治法（Divide and Conquer）求解特征值与特征向量；
   - 累乘特征向量矩阵 $Q = V Q_T$。
