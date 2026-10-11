# `linalg_eigh` Operator Specification

## 1. Mathematical Definition
Compute the eigenvalue decomposition of a real symmetric matrix $A \in \mathbb{R}^{n \times n}$:
$$A = Q \Lambda Q^T$$
where:
- $Q \in \mathbb{R}^{n \times n}$ is an orthogonal matrix: $Q^T Q = I$
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$ with eigenvalues sorted in ascending order:
  $$\lambda_1 \le \lambda_2 \le \dots \le \lambda_n$$

## 2. Shapes and Layouts
- **Inputs**:
  - `A`: `float32`, shape `[..., n, n]`, row-major.
- **Attributes**:
  - `UPLO`: `'L'` or `'U'`. If `'L'`, only the lower triangle of $A$ is used, and the upper triangle is assumed symmetric ($A_{i, j} = A_{j, i}$).
- **Outputs**:
  - `eigenvalues`: `float32`, shape `[..., n]`.
  - `eigenvectors`: `float32`, shape `[..., n, n]`.

## 3. Precision Acceptance Criteria
- Ground truth reference: `scipy.linalg.eigh(A_symm)` in `float64`.
- Criteria:
  1. Eigenvalues relative error:
     $$\max_{0 \le i < n} \frac{|\hat{\lambda}_i - \lambda_i|}{\max(1.0, |\lambda_i|)} \le 10^{-4}$$
  2. Orthogonality:
     $$\|\hat{Q}^T \hat{Q} - I\|_F \le 10^{-4}$$
  3. Reconstruction error:
     $$\frac{\|A - \hat{Q} \hat{\Lambda} \hat{Q}^T\|_F}{\|A\|_F} \le 10^{-4}$$
