#!/usr/bin/env python3
"""Local reference validation script for linalg_eigh."""
import numpy as np

def generate_case(n=32, batch_size=2, uplo='L', seed=42):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(batch_size):
        M = rng.standard_normal((n, n)).astype(np.float32)
        if uplo == 'L':
            A_symm = np.tril(M) + np.tril(M, -1).T
        else:
            A_symm = np.triu(M) + np.triu(M, 1).T

        # Ground truth using float64
        evals_ref, evecs_ref = np.linalg.eigh(A_symm.astype(np.float64))
        cases.append((M, A_symm, evals_ref.astype(np.float32), evecs_ref.astype(np.float32)))
    return cases

def verify_solution(A_symm, evals_pred, evecs_pred, evals_ref):
    n = A_symm.shape[0]
    # 1. Eigenvalues relative error
    eval_err = np.max(np.abs(evals_pred - evals_ref) / np.maximum(1.0, np.abs(evals_ref)))
    # 2. Orthogonality
    orth_err = np.linalg.norm(evecs_pred.T @ evecs_pred - np.eye(n))
    # 3. Reconstruction error
    recon = evecs_pred @ np.diag(evals_pred) @ evecs_pred.T
    norm_A = np.linalg.norm(A_symm)
    rec_err = np.linalg.norm(A_symm - recon) / max(1e-12, norm_A)

    passed = (eval_err <= 1e-4) and (orth_err <= 1e-4) and (rec_err <= 1e-4)
    return passed, eval_err, orth_err, rec_err

if __name__ == "__main__":
    cases = generate_case()
    print(f"Generated {len(cases)} test cases.")
    for i, (M, A_symm, evals_ref, evecs_ref) in enumerate(cases):
        passed, ev_err, orth_err, rec_err = verify_solution(A_symm, evals_ref, evecs_ref, evals_ref)
        print(f"Case {i+1}: passed={passed}, eval_err={ev_err:.2e}, orth_err={orth_err:.2e}, rec_err={rec_err:.2e}")
