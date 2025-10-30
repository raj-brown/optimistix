<h1 align='center'>SS-Quasi-Newton-Optimistix</h1>

This library includes two quasi-Newton optimizer i.e., SSBFGS and SSBroyden and  developed on top of Optimistix, which is a [JAX](https://github.com/google/jax) library for nonlinear solvers: root finding, minimisation, fixed points, and least squares.


## Installation

```bash
pip install git+https://github.com/raj-brown/optimistix.git
```

Requires Python 3.10+ and JAX 0.4.38+ and [Equinox](https://github.com/patrick-kidger/equinox) 0.11.11+.

## Documentation

Available at [https://docs.kidger.site/optimistix](https://docs.kidger.site/optimistix).

## Quick example

```python
import jax.numpy as jnp
import optimistix as optx
y0 = jnp.array(0.0)

def fn(y, _):
        return 0.5 * (y - jnp.tanh(y + 1)) ** 2

solver = optx.SSBFGS(rtol=1e-5, atol=1e-5)
solver = optx.BestSoFarMinimiser(solver)
sol = optx.minimise(fn, solver, jnp.array(0.0))
assert jnp.allclose(sol.value, 0.96118069, rtol=1e-5, atol=1e-5)
print("Assertion was successful!")
```

## Citation

If you found this library to be useful in academic work, then please cite: ([Journal Paper][https://www.sciencedirect.com/science/article/pii/S0045782525005808])


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

Also for Optimistix library please cite: ([arXiv link](https://arxiv.org/abs/2402.09983))

```bibtex
@article{optimistix2024,
    title={Optimistix: modular optimisation in JAX and Equinox},
    author={Jason Rader and Terry Lyons and Patrick Kidger},
    journal={arXiv:2402.09983},
    year={2024},
}
```



## Credit

Optimistix was primarily built by Jason Rader (@packquickly): [Twitter](https://twitter.com/packquickly); [GitHub](https://github.com/packquickly); [Website](https://www.packquickly.com/).


[https://www.sciencedirect.com/science/article/pii/S0045782525005808]: https://www.sciencedirect.com/science/article/pii/S0045782525005808