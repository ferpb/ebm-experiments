"""
Maximum likelihood training of a EBM on 2D data.

Th partition function Z is approximated by discretizing a bounded 2D grid
and summing exp(-E(x)) * dx, making MLE exact (up to grid resolution) without
contrastive divergence or MCMC during training.
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


def sample_data(key, n):
    key1, key2 = jax.random.split(key, 2)
    u = jax.random.uniform(key1, (n,))
    t = u * 4 * jnp.pi
    s = u + 0.1 * jax.random.uniform(key2, (n,))
    x = jnp.cos(t) * s
    y = jnp.sin(t) * s
    out = jnp.stack([x, y], axis=1)
    return out


def init_mlp(key, dims):
    params = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (d_out, d_in)) * jnp.sqrt(2.0 / d_in)
        b = jnp.zeros(d_out)
        params.append((W, b))
    return params


def energy(params, x):
    h = x
    for W, b in params[:-1]:
        h = jax.nn.swish(h @ W.T + b)
    W, b = params[-1]
    return (h @ W.T + b).squeeze(-1)


dims = [2, 32, 32, 1]

xs = jnp.linspace(-2, 2, 100)
ys = jnp.linspace(-2, 2, 100)

xx, yy = jnp.meshgrid(xs, ys)
all_x = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
dx = (xs[1] - xs[0]) * (ys[1] - ys[0])


def log_p(params, x):
    log_Z = jax.nn.logsumexp(-energy(params, all_x)) + jnp.log(dx)
    return -energy(params, x) - log_Z


key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
params = init_mlp(k1, dims)
data = sample_data(k2, 1000)

flat, unravel = ravel_pytree(params)

neg_ll = jax.jit(lambda flat: -log_p(unravel(flat), data).mean())
neg_ll_grad = jax.jit(jax.grad(neg_ll))

result = minimize(
    lambda params: float(neg_ll(jnp.array(params))),
    np.array(flat),
    jac=lambda x: np.array(neg_ll_grad(jnp.array(x))),
    method="L-BFGS-B",
    options={"maxiter": 500},
)

params = unravel(result.x)

all_log_probs = log_p(params, all_x)
all_probs = jnp.exp(all_log_probs)

print(f"log p(data): {log_p(params, data).mean()}")


def langevin_chain(score_fn, key, n_steps=2000, step_size=0.001):
    key, init_key = jax.random.split(key)
    x0 = jax.random.normal(init_key, (2,))

    def step(x, key):
        noise = jax.random.normal(key, (2,))
        score = score_fn(x)
        return x + step_size * score + jnp.sqrt(2 * step_size) * noise, None

    keys = jax.random.split(key, n_steps)
    x_final, _ = jax.lax.scan(step, x0, keys)
    return x_final


key, k = jax.random.split(key)
score_fn = jax.grad(lambda x: log_p(params, x[None]).squeeze())
chain_keys = jax.random.split(k, 1000)
samples = jax.vmap(langevin_chain, in_axes=(None, 0))(score_fn, chain_keys)


plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(data[:, 0], data[:, 1], s=2)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title("data")

plt.subplot(132)
plt.contourf(xx, yy, all_probs.reshape(100, 100))
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title("p(x)")

plt.subplot(133)
plt.scatter(samples[:, 0], samples[:, 1], s=2)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title("samples")

plt.savefig("ebm_exact_mle.png")
