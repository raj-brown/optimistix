import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from optimistix._solver.dsgnar import (
    _apply_srct,
    _lift_srct,
    _make_srct,
)


def test_dsgnar_import_and_run():
    def residual(x, args):
        del args
        return jnp.array([x[0] - 2.0, 2.0 * x[0] - 4.0])

    x0 = jnp.array([0.0])

    solver = optx.DSGNAR(rtol=1e-6, atol=1e-6, sketch_size=2)
    sol = optx.least_squares(
        residual,
        solver,
        x0,
        options={"key": jax.random.PRNGKey(0)},
        max_steps=2,
        throw=False,
    )

    assert sol.result == optx.RESULTS.successful
    assert sol.value.shape == x0.shape
    assert jnp.isfinite(sol.stats["final_loss"])


def test_srct_preserves_step_norm():
    key = jax.random.PRNGKey(123)

    n_params = 32
    sketch_size = 8

    key_srct, key_z = jax.random.split(key)

    srct = _make_srct(
        key_srct,
        n_params,
        sketch_size,
        jnp.float32,
    )

    z = jax.random.normal(
        key_z,
        (sketch_size,),
    )

    p = _lift_srct(
        z,
        srct,
        n_params,
    )

    assert p.shape == (n_params,)

    np.testing.assert_allclose(
        jnp.linalg.norm(p),
        jnp.linalg.norm(z),
        rtol=1e-5,
        atol=1e-6,
    )


def test_srct_matrix_application_matches_explicit_embedding():
    key = jax.random.PRNGKey(456)

    n_params = 32
    sketch_size = 8
    n_rows = 11

    key_srct, key_J = jax.random.split(key)

    srct = _make_srct(
        key_srct,
        n_params,
        sketch_size,
        jnp.float32,
    )

    J = jax.random.normal(
        key_J,
        (n_rows, n_params),
    )

    # --------------------------------------------------------
    # Fast implementation
    # --------------------------------------------------------

    J_sketch = _apply_srct(
        J,
        srct,
    )

    # --------------------------------------------------------
    # Explicitly construct Q = Omega S
    #
    # This is ONLY for testing.
    # We will never construct Q in the real solver.
    # --------------------------------------------------------

    eye_s = jnp.eye(
        sketch_size,
        dtype=J.dtype,
    )

    Q_columns = jax.vmap(
        lambda e: _lift_srct(
            e,
            srct,
            n_params,
        )
    )(eye_s)

    # vmap returns:
    #
    #     (sketch_size, n_params)
    #
    # We want Q:
    #
    #     (n_params, sketch_size)

    Q = Q_columns.T

    J_sketch_explicit = J @ Q

    assert J_sketch.shape == (
        n_rows,
        sketch_size,
    )

    np.testing.assert_allclose(
        J_sketch,
        J_sketch_explicit,
        rtol=2e-5,
        atol=2e-5,
    )


def test_srct_jit():
    key = jax.random.PRNGKey(789)

    n_params = 32
    sketch_size = 8

    key_srct, key_J = jax.random.split(key)

    srct = _make_srct(
        key_srct,
        n_params,
        sketch_size,
        jnp.float32,
    )

    J = jax.random.normal(
        key_J,
        (10, n_params),
    )

    apply_jit = jax.jit(lambda x: _apply_srct(x, srct))

    result = apply_jit(J)

    assert result.shape == (10, sketch_size)

    assert jnp.all(jnp.isfinite(result))
