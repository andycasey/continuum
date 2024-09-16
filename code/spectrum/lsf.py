import numpy as np
from scipy import sparse, stats
from typing import Sequence, Optional

_fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))


def instrument_lsf_kernel(
    λi: Sequence[float],
    λo: Sequence[float],
    Ro: float,
    Ri: Optional[float] = np.inf,
    **kwargs
):
    """
    Construct a sparse matrix to convolve fluxes at input wavelengths (λi) at an instrument spectral
    resolution (R) and resample to the given output wavelengths (λo).

    :param λi:
        A N-length array of input wavelength values.

    :param λo:
        A M-length array of output wavelength values.

    :param Ro:
        Spectral resolution.

    :param Ri: [optional]
        Input spectral resolution (default: infinity).

    :returns:
        A (N, M) sparse array representing a convolution kernel.
    """
    R_inv = np.sqrt(float(Ro) ** (-2) - float(Ri) ** (-2))
    data, i, j = ([], [], [])
    for ii, λ in enumerate(λo):
        si, ei, ϕ = _instrument_lsf_kernel(λi, λ, R_inv, **kwargs)
        i.extend([ii] * (ei - si))
        j.extend(range(si, ei))
        data.extend(ϕ)
    return sparse.coo_matrix((data, (j, i)), shape=(λi.size, λo.size))


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
    for i, λ_i in enumerate(λ):
        n_pix = int(np.ceil(λ_i * scale))
        si, ei = (max(0, i - n_pix), min(i + n_pix + 1, N))  # ignoring edge effects

        λ_delta_max = λ_i * vsini_c
        λ_delta = λ[si:ei] - λ_i
        λ_ratio_sq = (λ_delta / λ_delta_max) ** 2.0
        ϕ = c1 * np.sqrt(1.0 - λ_ratio_sq) + c2 * (1.0 - λ_ratio_sq)
        ϕ[λ_ratio_sq >= 1.0] = 0.0  # flew too close to the sun
        ϕ /= np.sum(ϕ)

        data.extend(ϕ)
        row_index.extend(list(range(si, ei)))
        col_index.extend([i] * (ei - si))

    return sparse.csr_matrix((data, (row_index, col_index)), shape=(λ.size, λ.size))


def _lsf_sigma(λ: float, R_inv: float):
    """
    Return the Gaussian width for the line spread function at wavelength λ with spectral resolution R.

    :param λ:
        Wavelength to evaluate the LSF.

    :param R_inv:
        The inverse of the relative spectral resolution (1/R), after accounting for the input spectral
        resolution:

            R_inv = sqrt(Ro**(-2) - Ri**(-2))

    :returns:
        The width of a Gaussian to represent the width of the line spread function.
    """
    return λ * _fwhm_to_sigma * R_inv


def _lsf_sigma_and_bounds(λo: float, R_inv: float, σ_window: Optional[float] = 5):
    """
    Return the Gaussian width for the line spread function, and the lower and upper bounds where the LSF contributes.

    :param λo:
        Wavelength to evaluate the LSF.

    :param R:
        The output spectral resolution, relative to any input resolution.

    :param σ_window: [optional]
        The number of sigma where the LSF contributes (default: 5).

    :returns:
        A three-length tuple containing: the LSF sigma, and the lower and upper wavelengths where the LSF contributes.
    """
    σ = _lsf_sigma(λo, R_inv)
    return (σ, λo - σ_window * σ, λo + σ_window * σ)


def _instrument_lsf_kernel(
    λi: Sequence[float], λo: Sequence[float], R_inv: float, **kwargs
):
    """
    Calculate the convolution kernel for the given instrument line spread function at the wavelengths specified,
    centered on the given central wavelength.

    :param λi:
        The input wavelength array.

    :param λo:
        Output wavelength array to evaluate the LSF on.

    :param R:
        Spectral resolution.

    :returns:
        A two-length tuple containing the mask where the LSF contributes, and the normalised kernel.
    """
    σ, lower, upper = _lsf_sigma_and_bounds(λo, R_inv, **kwargs)
    si, ei = np.searchsorted(λi, [lower, upper])
    ϕ = stats.norm.pdf(λi[si:ei], loc=λo, scale=σ)
    ϕ /= np.sum(ϕ)
    return (si, ei, ϕ)
