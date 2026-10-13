"""SSBFGS / SSBroyden must compute b_k with the step length actually taken.

`b_k = -alpha_k s_k^T g_k / s_k^T y_k` (Kiyani et al. 2025, Eq. (5)) needs the step
length `alpha_k` of the accepted step. The `step_size` argument of `update_hessian` is
the line search's proposal for the *next* step (1.0 after acceptance for
`BacktrackingArmijo` and for `Zoom(initial_guess_strategy="one")`), so it cannot be
used for this.
"""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest
from optimistix._search import FunctionInfo
from optimistix._solver.quasi_newton import _taken_step_length


jax.config.update("jax_enable_x64", True)


def test_taken_step_length_recovers_alpha():
    H = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    g = jnp.array([0.7, -1.1])
    p = -H @ g
    op = lx.MatrixLinearOperator(H, lx.positive_semidefinite_tag)
    f_inv = FunctionInfo.EvalGradHessianInv(jnp.array(0.0), g, op)
    assert jnp.allclose(_taken_step_length(0.37 * p, g, f_inv, True), 0.37)
    B = lx.MatrixLinearOperator(jnp.linalg.inv(H), lx.positive_semidefinite_tag)
    f_dir = FunctionInfo.EvalGradHessian(jnp.array(0.0), g, B)
    assert jnp.allclose(_taken_step_length(0.37 * p, g, f_dir, False), 0.37)


def _ill_conditioned_quadratic(n=20, cond=1e3, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = jnp.asarray(q @ np.diag(np.logspace(0, np.log10(cond), n)) @ q.T)
    return (lambda x, _: 0.5 * x @ a @ x), jnp.asarray(rng.standard_normal(n))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("solver_cls", [optx.SSBFGS, optx.SSBroyden])
def test_converges_on_ill_conditioned_quadratic(solver_cls, seed):
    # With alpha_k taken as 1, SSBroyden (default backtracking search) stalls here.
    fn, x0 = _ill_conditioned_quadratic(seed=seed)
    solver = solver_cls(rtol=1e-12, atol=1e-12, use_inverse=True)
    sol = optx.minimise(fn, solver, x0, max_steps=500, throw=False)
    assert fn(sol.value, None) < 1e-16
