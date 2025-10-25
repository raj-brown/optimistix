<h1 align='center'>SS-Quasi-Newton-Optimistix</h1>

This library includes two quasi-Newton optimizer i.e., SSBFGS and SSBroyden and  developed on top of Optimistix, which is a [JAX](https://github.com/google/jax) library for nonlinear solvers: root finding, minimisation, fixed points, and least squares.


## Installation

```bash
pip install https://github.com/raj-brown/optimistix.git
```

Requires Python 3.10+ and JAX 0.4.38+ and [Equinox](https://github.com/patrick-kidger/equinox) 0.11.11+.

## Documentation

Available at [https://docs.kidger.site/optimistix](https://docs.kidger.site/optimistix).

## Quick example

```python
import jax.numpy as jnp
import optimistix as optx

# Let's solve the ODE dy/dt=tanh(y(t)) with the implicit Euler method.
# We need to find y1 s.t. y1 = y0 + tanh(y1)dt.

y0 = jnp.array(1.)
dt = jnp.array(0.1)

def fn(y, args):
    return y0 + jnp.tanh(y) * dt

solver = optx.Newton(rtol=1e-5, atol=1e-5)
sol = optx.fixed_point(fn, solver, y0)
y1 = sol.value  # satisfies y1 == fn(y1)
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