<h1 align="center">SS-Quasi-Newton-Optimistix</h1>

This library provides two quasi-Newton optimizers — **SSBFGS** and **SSBroyden** — developed on top of [Optimistix](https://github.com/patrick-kidger/optimistix), a [JAX](https://github.com/google/jax)-based library for nonlinear solvers including root finding, minimization, fixed-point problems, and least-squares optimization.

---

# Installation

The package requires Python 3.11 or newer. A virtual environment is recommended.

## Install from the main branch

```bash
git clone https://github.com/raj-brown/optimistix.git
cd optimistix
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,tests]'
```

Requires Python 3.11+. This installs the package in editable mode together with the
development tools and test dependencies. The default JAX installation uses the CPU
backend.

## Install the published package

```bash
python -m pip install optimistix
```

## Optional CUDA installation

For NVIDIA CUDA, install the JAX wheel that matches your CUDA version by following the
official [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html),
then install this package:

```bash
python -m pip install optimistix
```

## Verify the installation

```bash
python -c "import jax, optimistix; print(jax.devices()); print(optimistix.__file__)"
```

---

# Documentation

The Optimistix documentation is available at:

👉 https://docs.kidger.site/optimistix

---

# Quick Example

```python
import jax.numpy as jnp
import optimistix as optx


def fn(y, _):
    return 0.5 * (y - jnp.tanh(y + 1)) ** 2


solver = optx.SSBFGS(rtol=1e-5, atol=1e-5)
solver = optx.BestSoFarMinimiser(solver)

sol = optx.minimise(fn, solver, jnp.array(0.0))

assert jnp.allclose(sol.value, 0.96118069, rtol=1e-5, atol=1e-5)
print("Assertion was successful!")
```

---

# Citation

If you found this library useful in academic research, please cite the following works.

## SSBFGS / SSBroyden Paper

```bibtex
@article{kiyani2025optimizing,
  title={Optimizing the optimizer for physics-informed neural networks and Kolmogorov-Arnold networks},
  author={Kiyani, Elham and Shukla, Khemraj and Urb{\'a}n, Jorge F and Darbon, J{\'e}r{\^o}me and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={446},
  pages={118308},
  year={2025},
  publisher={Elsevier}
}
```

Paper link:  
https://www.sciencedirect.com/science/article/pii/S0045782525005808

---

## Curvature-aware optimization for high-accuracy physics-informed neural networks

```bibtex
@article{jnini2026curvature,
  title={Curvature-aware optimization for high-accuracy physics-informed neural networks},
  author={Jnini, Anas and Kiyani, Elham and Shukla, Khemraj and Urban, Jorge F and Daryakenari, Nazanin Ahmadi and Muller, Johannes and Zeinhofer, Marius and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2604.05230},
  year={2026}
}
```

---

## Optimistix

Please also cite the original Optimistix library:

```bibtex
  title={Optimistix: modular optimisation in JAX and Equinox},
  author={Rader, Jason and Lyons, Terry and Kidger, Patrick},
  journal={arXiv:2402.09983},
Optimistix is also co-maintained by Johanna Haffner (@johannahaffner):
[GitHub](https://github.com/johannahaffner); [Website](https://haffner.dev).
arXiv: https://arxiv.org/abs/2402.09983

---

# Credits

Optimistix was primarily developed by Jason Rader (@packquickly).

- GitHub: https://github.com/packquickly
- Website: https://www.packquickly.com/
- Twitter/X: https://twitter.com/packquickly
Optimistix is also co-maintained by Johanna Haffner (@johannahaffner):
[GitHub](https://github.com/johannahaffner); [Website](https://haffner.dev).
