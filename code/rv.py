#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Functions related to radial velocity measurement and correction. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["cross_correlate"]

import logging
import numpy as np
from scipy import interpolate
from scipy.optimize import leastsq


logger = logging.getLogger(__name__)


def cross_correlate(observed_λ, observed_flux, observed_ivar, template_λ, template_flux, λ_range=None,
    use_weight=False, apodize=0, resample="template", verbose=False):
    """
    Cross-correlate the observed spectrum against a rest-frame template spectrum
    and measure the radial velocity of the source.

    :param observed_spectrum:
        The observed spectrum.

    :type observed_spectrum:
        :class:`specutils.Spectrum1D`

    :param template_spectrum:
        A rest-frame template spectrum.

    :type template_spectrum:
        :class:`specutils.Spectrum1D`

    :param λ_range: [optional]
        A two-length tuple containing the start and end wavelengths to consider.
        If not provided, then the overlap of the `observed_spectrum` and the
        `template_spectrum` will be used.

    :param apodize: [optional]
        The fraction of pixels to apodize on either side of the spectrum.

    :param resample: [optional]
        Whether to resample the 'template' onto the observed spectrum, or
        resample the 'observed' spectrum onto the template.

    :returns:
        The radial velocity, uncertainty in radial velocity, and the CCF.
    """


    if λ_range is None:
        # Use the common ranges.
        λ_range = (
            np.max([
                observed_λ[0],
                template_λ[0]
            ]),
            np.min([
                observed_λ[-1],
                template_λ[-1]
            ])
        )

    if not isinstance(λ_range, (tuple, list, np.ndarray)) \
    or len(λ_range) != 2:
        raise TypeError("wavelength region must be a two length list-type")

    if apodize != 0:
        raise NotImplementedError("apodization not implemented yet")
        
    resample = resample.lower()
    if resample == "template":
        idx = np.searchsorted(observed_λ, λ_range)
        finite = np.isfinite(observed_flux[idx[0]:idx[1]])

        λ = observed_λ[idx[0]:idx[1]][finite]
        observed_flux = observed_flux[idx[0]:idx[1]][finite]
        observed_ivar = observed_ivar[idx[0]:idx[1]][finite]

        func = interpolate.interp1d(
            template_λ, template_flux,
            bounds_error=False, fill_value=0.0)
        template_flux = func(λ)

    elif resample == "observed":
        raise NotImplementedError("why would you do this?")

    else:
        raise ValueError("resample must be 'template' or 'observed'")


    # Perform the cross-correlation
    padding = observed_flux.size + template_flux.size
    # Is this necessary?: # TODO
    x_norm = observed_flux - np.mean(observed_flux[np.isfinite(observed_flux)])
    y_norm = template_flux - np.mean(template_flux[np.isfinite(template_flux)])
    if use_weight:
        x_norm = x_norm * observed_ivar


    Fx = np.fft.fft(x_norm, padding, )
    Fy = np.fft.fft(y_norm, padding, )
    iFxy = np.fft.ifft(Fx.conj() * Fy).real

    ccf_tt = np.fft.ifft(Fy.conj() * Fy).real
    varxy = np.sqrt(np.inner(x_norm, x_norm) * np.inner(y_norm, y_norm))

    if use_weight:
        fft_result = iFxy
    else:
        fft_result = iFxy/varxy

    # Put around symmetry axis.
    num = len(fft_result) - 1 if len(fft_result) % 2 else len(fft_result)

    fft_y = np.zeros(num)
    fft_y[:num//2] = fft_result[num//2:num]
    fft_y[num//2:] = fft_result[:num//2]

    fft_tt = np.zeros(num)
    fft_tt[:num//2] = ccf_tt[num//2:num]
    fft_tt[num//2:] = ccf_tt[:num//2]

    fft_x = np.arange(num) - num/2

    # Get initial guess of peak.
    p0 = np.array([fft_x[np.argmax(fft_y)], np.max(fft_y), 10])

    gaussian = lambda p, x: p[1] * np.exp(-(x - p[0])**2 / (2.0 * p[2]**2))
    errfunc = lambda p, x, y: y - gaussian(p, x)

    try:
        p1, ier = leastsq(errfunc, p0.copy(), args=(fft_x, fft_y))

    except:
        logger.exception("Exception in measuring peak of CCF:")
        raise

    si = num//2 - 50
    ei = num//2 + 50
    p_tt, ier2 = leastsq(errfunc, [0, np.max(fft_tt), 5], args=(fft_x[si:ei], fft_tt[si:ei]))

    # Create functions for interpolating back onto the λ map
    fft_points = (0, p1[0], p1[2])
    interp_x = np.arange(num/2) - num/4

    wl_points = []
    for point in fft_points:
        idx = np.searchsorted(interp_x, point)
        try:
            f = interpolate.interp1d(interp_x[idx-3:idx+3], λ[idx-3:idx+3],
                bounds_error=True, kind='cubic')
        except ValueError as e:
            print("Interpolation error! Probably bad template? Returning nans with raw CCF")
            print(e)
            print(fft_points, point)
            return np.nan, np.nan, np.array([fft_x, fft_y])
        wl_points.append(f(point))

    # Calculate velocity 
    c = 299792458e-3 # km/s
    f, g, h = wl_points
    rv = c * (1 - g/f)

    # Create a CCF spectrum.
    ccf = np.array([fft_x * (rv/p1[0]), fft_y])

    # Calculate uncertainty
    if use_weight:
        # The ccf should be normalized so that it is close to a chi2 distribution
        # Reducing the ccf by 0.5 from its peak value gives 1-sigma error
        # We approximate this assuming the Gaussian fit is correct
        ymax = p1[1]
        minfunc = lambda x: (ymax - 0.5) - gaussian(p1, x) 
        try:
            xerr, ier = leastsq(minfunc, p1[0] + p1[2])
        except:
            logger.exception("Exception in measuring 1sigma offset for CCF")
            raise
        point = np.abs(xerr - p1[0])
        idx = np.searchsorted(interp_x, point)
        try:
            func = interpolate.interp1d(interp_x[idx-3:idx+3], λ[idx-3:idx+3],
                bounds_error=True, kind='cubic')
        except ValueError as e:
            print("Interpolation error in solving for error!")
            print(e, point)
            return np.nan, np.nan, np.array([fft_x, fft_y])
        h = func(point)[0]
        if verbose: print(f,g,h, (h-f)/g, ymax)
        rv_uncertainty = np.abs(c * (h-f)/g)
    else:
        # Approx Uncertainty based on simple gaussian
        # This is not a real uncertainty as it doesn't take into account data errors
        rv_uncertainty = np.abs(c * (h-f)/g)

    return (rv, rv_uncertainty, ccf, np.clip(p1[2] - p_tt[2], 0, np.inf))



def measure_order_velocities(orders, template, norm_kwargs, **kwargs):
    """
    Run cross correlation against a list of orders
    Return Nx3 array, where columns are order_num, rv, e_rv, wlmin, wlmax
    """
    N = len(orders)
    rv_output = np.zeros((N,5))
    for i, order in enumerate(orders):
        normorder = order.fit_continuum(**norm_kwargs)
        try:
            rv, e_rv, ccf = cross_correlate(normorder, template, **kwargs)
        except:
            rv, e_rv = np.nan, np.nan
        try:
            order_num = order.metadata["ECORD{}".format(i)]
        except:
            order_num = i
        rv_output[i,0] = order_num
        rv_output[i,1] = rv
        rv_output[i,2] = e_rv
        rv_output[i,3] = np.min(order.λ)
        rv_output[i,4] = np.max(order.λ)
    return rv_output
