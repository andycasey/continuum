from typing import Sequence
import jax
from jax import numpy as jnp
import numpy as np
#from scipy import sparsefrom jax.e
from jax.experimental.sparse import BCOO

def rotational_broadening_kernel(λ: Sequence[float], vsini: float, epsilon: float):
    """
    Construct a sparse matrix to convolve fluxes at input wavelengths (λ) with a rotational broadening kernel
    with a given vsini and epsilon.

    :param λ:
        A N-length array of input wavelength values.

    :param vsini:
        The projected rotational velocity of the star in km/s.

    :param epsilon:
        The limb darkening coefficient.

    :returns:
        A (N, N) sparse array representing a convolution kernel.
    """

    # Let's pre-calculate some things that are needed in the hot loop.
    denominator = np.pi * vsini * (1.0 - epsilon / 3.0)
    c1 = 2.0 * (1.0 - epsilon) / denominator
    c2 = 0.5 * np.pi * epsilon / denominator

    vsini_c = vsini / 299792.458
    scale = vsini_c / (λ[1] - λ[0])  # assume uniform sampling
    N = λ.size

    data, row_index, col_index = ([], [], [])
    indices = []
    for i, λ_i in enumerate(λ):
        n_pix = int(jnp.ceil(λ_i * scale))
        si, ei = (max(0, i - n_pix), min(i + n_pix + 1, N))  # ignoring edge effects

        λ_delta_max = λ_i * vsini_c
        λ_delta = λ[si:ei] - λ_i
        #λ_ratio_sq = jnp.clip((λ_delta / λ_delta_max) ** 2.0, 0, 1 - 1e-5) # check this
        λ_ratio_sq = (λ_delta / λ_delta_max) ** 2.0
        assert np.all(1 >= λ_ratio_sq)
        
        # approximation of sqrt(1 - x^2)
        
        
        ϕ = c1 * jnp.sqrt(1.0 - λ_ratio_sq) + c2 * (1.0 - λ_ratio_sq)
        #ϕ[λ_ratio_sq >= 1.0] = 0.0  # flew too close to the sun 
        ϕ /= jnp.sum(ϕ)
        data.extend(ϕ)
        indices.extend([(ii, jj) for ii, jj in zip(range(si, ei), [i] * (ei - si))])

    #return BCOO((data, indices), shape=(λ.size, λ.size))
    return sparse.csr_matrix((data, (row_index, col_index)), shape=(λ.size, λ.size))
    

np.random.seed(0)
y = np.random.uniform(0, 1, 8575)
λ = 10**(4.179 + 6e-6 * np.arange(8575))

#from scipy.ndimage import gaussian_filter1d
from jax_ndimage import gaussian_filter1d

def f(θ):
    #K = rotational_broadening_kernel(λ, θ[0], 0.6)
    #return K @ y
    return gaussian_filter1d(y, θ[0])
    

g = jax.jacfwd(f)


    