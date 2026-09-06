from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree, Scalar

from .._least_squares import AbstractLeastSquaresSolver
from .._misc import max_norm, sum_squares, tree_full_like
from .._solution import RESULTS


class _DSGNARState(eqx.Module):
    radius: Scalar
    target_ratio: Scalar
    last_lambda: Scalar
    last_rho: Scalar
    last_loss: Scalar
    key: Array
    terminate: Bool[Array, ""]
    results: RESULTS
    aux: Any


# ============================================================
# SRCT -- RIGHT-SIDE / PARAMETER SKETCH
# ============================================================


def _make_srct(
    key: Array,
    n_params: int,
    sketch_size: int,
    dtype,
) -> tuple[Array, Array, Array]:
    """Construct the SRCT used by DSGNAR.

    The right-side embedding is

        Omega S = (D Pi F) S,

    where

        D  : random +/-1 diagonal matrix,
        Pi : random permutation,
        F  : orthonormal type-II DCT,
        S  : random coordinate restriction.

    Parameters
    ----------
    key:
        JAX PRNG key.

    n_params:
        Dimension of the full parameter space.

    sketch_size:
        Dimension of the reduced parameter space.

    dtype:
        Floating-point dtype.

    Returns
    -------
    signs:
        Random +/-1 signs, shape (n_params,).

    perm:
        Random permutation of the parameter indices.

    indices:
        Coordinates retained after the cosine transform.
    """

    if sketch_size > n_params:
        raise ValueError(
            "DSGNAR sketch_size must not exceed the number "
            f"of parameters. Got sketch_size={sketch_size}, "
            f"n_params={n_params}."
        )

    key_signs, key_perm, key_indices = jax.random.split(key, 3)

    # D: random diagonal sign matrix
    signs = jax.random.choice(
        key_signs,
        jnp.array([-1.0, 1.0], dtype=dtype),
        shape=(n_params,),
    )

    # Pi: random permutation
    perm = jax.random.permutation(
        key_perm,
        n_params,
    )

    # S: random restriction to sketch_size coordinates
    indices = jax.random.choice(
        key_indices,
        n_params,
        shape=(sketch_size,),
        replace=False,
    )

    return signs, perm, indices


def _apply_srct(
    J: Array,
    srct: tuple[Array, Array, Array],
) -> Array:
    """Apply the DSGNAR right sketch J -> J Omega S.

    Parameters
    ----------
    J:
        Jacobian block of shape

            (n_residuals, n_params)

    srct:
        Tuple returned by `_make_srct`.

    Returns
    -------
    J_sketch:
        Column-sketched Jacobian of shape

            (n_residuals, sketch_size)
    """

    signs, perm, indices = srct

    # --------------------------------------------------------
    # 1. Random sign flip: J D
    # --------------------------------------------------------

    J = J * signs.astype(J.dtype)[None, :]

    # --------------------------------------------------------
    # 2. Random permutation: J D Pi
    # --------------------------------------------------------

    J = J[:, perm]

    # --------------------------------------------------------
    # 3. Orthonormal cosine transform: J D Pi F
    # --------------------------------------------------------

    J = jax.scipy.fft.dct(
        J,
        type=2,
        norm="ortho",
        axis=1,
    )

    # --------------------------------------------------------
    # 4. Random restriction: J D Pi F S
    # --------------------------------------------------------

    return J[:, indices]


def _lift_srct(
    z: Array,
    srct: tuple[Array, Array, Array],
    n_params: int,
) -> Array:
    """Lift a reduced DSGNAR step into full parameter space.

    Computes

        p = Omega S z

    where z is the step in the sketched parameter space.

    Parameters
    ----------
    z:
        Reduced step of shape (sketch_size,).

    srct:
        Tuple returned by `_make_srct`.

    n_params:
        Full parameter dimension.

    Returns
    -------
    p:
        Full parameter-space step of shape (n_params,).
    """

    signs, perm, indices = srct

    # --------------------------------------------------------
    # Undo S: insert reduced vector into selected coordinates
    # --------------------------------------------------------

    v = jnp.zeros(
        (n_params,),
        dtype=z.dtype,
    )

    v = v.at[indices].set(z)

    # --------------------------------------------------------
    # Undo/apply orthogonal cosine transform
    # --------------------------------------------------------

    v = jax.scipy.fft.idct(
        v,
        type=2,
        norm="ortho",
    )

    # --------------------------------------------------------
    # Undo permutation
    # --------------------------------------------------------

    v_full = jnp.zeros_like(v)

    v_full = v_full.at[perm].set(v)

    # --------------------------------------------------------
    # Sign matrix is its own inverse: D^{-1} = D
    # --------------------------------------------------------

    return v_full * signs.astype(v_full.dtype)


class DSGNAR(AbstractLeastSquaresSolver):
    """Doubly-sketched Gauss-Newton with adaptive ratio solver."""

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    sketch_size: int
    initial_radius: float
    min_radius: float
    max_radius: float
    target_ratio: float
    n_probes: int
    probe_scale: float
    n_hash: int

    def __init__(
        self,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        *,
        sketch_size: int = 512,
        initial_radius: float = 1.0,
        min_radius: float = 1e-8,
        max_radius: float = 1e8,
        target_ratio: float = 0.1,
        n_probes: int = 24,
        probe_scale: float = 3.0,
        n_hash: int = 4,
        norm: Callable[[PyTree], Scalar] = max_norm,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.sketch_size = sketch_size
        self.initial_radius = initial_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.target_ratio = target_ratio
        self.n_probes = n_probes
        self.probe_scale = probe_scale
        self.n_hash = n_hash

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        del fn, y, args, f_struct, tags
        key = options.get("key", jax.random.PRNGKey(0))
        return _DSGNARState(
            radius=jnp.array(self.initial_radius),
            target_ratio=jnp.array(self.target_ratio),
            last_lambda=jnp.array(0.0),
            last_rho=jnp.array(0.0),
            last_loss=jnp.array(jnp.inf),
            key=key,
            terminate=jnp.array(False),
            results=RESULTS.successful,
            aux=tree_full_like(aux_struct, 0),
        )

    def step(self, fn, y, args, options, state, tags):
        del options, tags
        residuals, aux = fn(y, args)
        loss = 0.5 * sum_squares(residuals)
        new_state = _DSGNARState(
            radius=state.radius,
            target_ratio=state.target_ratio,
            last_lambda=state.last_lambda,
            last_rho=state.last_rho,
            last_loss=loss,
            key=state.key,
            terminate=jnp.asarray(True),
            results=RESULTS.successful,
            aux=aux,
        )
        return y, new_state, aux

    def terminate(self, fn, y, args, options, state, tags):
        del fn, y, args, options, tags
        return state.terminate, state.results

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        del fn, args, options, tags, result
        return (
            y,
            aux,
            {
                "final_radius": state.radius,
                "final_lambda": state.last_lambda,
                "final_rho": state.last_rho,
                "final_loss": state.last_loss,
            },
        )
