
import numpy as np

import h5py as h5

import numpy as np
import gzip
import warnings
import pickle
import os
from time import time

from itertools import cycle
from functools import cache
from typing import Optional, Tuple, Sequence, Union
from scipy import optimize as op
from sklearn.decomposition._nmf import _fit_coordinate_descent
from sklearn.exceptions import ConvergenceWarning
from scipy import interpolate
from astropy.constants import c
from threading import Thread

from typing import Iterable


from spectrum.lsf import instrument_lsf_kernel, rotational_broadening_kernel
from spectrum import Spectrum
from scipy.optimize._lsq.trf_linear import trf_linear


warnings.simplefilter("ignore")

C_KM_S = c.to("km/s").value


import numpy as np



from rv import cross_correlate
from continuum_basis import ContinuumBasis, Sinusoids, Polynomial
from utils import cumulative


def apply_radial_velocity_shift(λ, v_rel):
    if v_rel == 0 or v_rel is None:
        return λ
    beta = v_rel / C_KM_S
    scale = np.sqrt((1 + beta) / (1 - beta))     
    return λ * scale


class Clam:
    
    """Constrained linear absorption model, with applications to echelle spectra."""
    
    def __init__(
        self, 
        stellar_vacuum_wavelength, 
        stellar_basis_vectors, 
        telluric_vacuum_wavelength=None,
        telluric_basis_vectors=None,
        meta=None,         
    ):
        """
        :param stellar_vacuum_wavelength:
            An array of vacuum wavelengths (in Angstroms) of the basis vectors.
        
        :param stellar_basis_vectors:
            A (C, P)-shape array of stellar basis vectors with non-negative entries. Here, C is
            the number of basis vectors and P is the same as the size of `stellar_vacuum_wavelength`.
                    
        :param telluric_basis_vectors: [optional]
            A (B, P)-shape array of telluric basis vectors with non-negative entries. Here, P is
            the same as the size of `stellar_vacuum_wavelength`.
            
        :param meta: [optional]
            A metadata dictionary.            
        """
        self.stellar_vacuum_wavelength = stellar_vacuum_wavelength
        self.stellar_basis_vectors = stellar_basis_vectors  
        self.telluric_vacuum_wavelength = telluric_vacuum_wavelength          
        self.telluric_basis_vectors = telluric_basis_vectors
        self.meta = meta or dict()   
        return None        
    

    @property
    def can_model_tellurics(self):
        return (
            self.telluric_basis_vectors is not None 
        and self.telluric_vacuum_wavelength is not None
        )
            

    def _prepare_basis_vectors(self, λ_observed, λ, basis_vectors, v_rel=None, vsini=None, R=None):


        λ_out = apply_radial_velocity_shift(λ, v_rel)
        
        kernels = []
        if vsini is not None:
            kernels.append(rotational_broadening_kernel(λ, vsini, 0.6))        
        if Ro is not None:
            kernels.append(instrument_lsf_kernel(λ, λ_out, R))

        raise a


    def resample_basis_vectors(self, λ_vacuum, vsini=None, stellar_v_rel=0.0, telluric_v_rel=0.0, Ro=None, Ri=None, telluric=False):
        """
        Resample (and optionally convolve) the basis vectors to the observed wavelengths.
        
        :param λ_vacuum:
            The observed vacuum wavelengths.
        
        :param Ro: [optional]
            The spectral resolution of the observations. If `None` is given, then no
            convolution will take place; the basis vectors will be interpolated.
        
        :param Ri: [optional]
            The spectral resolution of the basis vectors. If `None` is given then
            it defaults to the input spectral resolution stored in the metadata of
            the basis vector file, or infinity if the input spectral resolution is
            not stored in that file.
                
        """

        print(f"stellar_v_rel={stellar_v_rel}")
        
        λ_vacuum_obs = apply_radial_velocity_shift(λ_vacuum, stellar_v_rel)
        
        if Ro is None:
            if vsini is not None:
                K_lsf = rotational_broadening_kernel(self.stellar_vacuum_wavelength, vsini, 0.6)
                basis_vectors = [_interpolate_basis_vector(λ_vacuum_obs, self.stellar_vacuum_wavelength, self.stellar_basis_vectors @ K_lsf)]
                assert not telluric

            else:
                # Interpolation only.            
                basis_vectors = [_interpolate_basis_vector(λ_vacuum_obs, self.stellar_vacuum_wavelength, self.stellar_basis_vectors)]
                if telluric:
                    basis_vectors.append(_interpolate_basis_vector(λ_vacuum, self.telluric_vacuum_wavelength, self.telluric_basis_vectors))
            
        else:
            # Convolution
            Ri = Ri or self.meta.get("Ri", np.inf)

            # special case to avoid double-building the same convolution kernel            
            K = instrument_lsf_kernel(self.stellar_vacuum_wavelength, λ_vacuum_obs, Ro, Ri)                

            basis_vectors = [self.stellar_basis_vectors @ K]
            if telluric:
                # Only reconstruct the kernel if the v_rel != 0
                if stellar_v_rel != 0 or telluric_v_rel != 0:
                    K = instrument_lsf_kernel(
                        self.telluric_vacuum_wavelength, 
                        apply_radial_velocity_shift(λ_vacuum, telluric_v_rel),
                        Ro, 
                        Ri
                    )
                basis_vectors.append(self.telluric_basis_vectors @ K)
    
        basis_vectors = np.vstack(basis_vectors)
        S = self.stellar_basis_vectors.shape[0]
        T = basis_vectors.shape[0] - S

        return (basis_vectors, S, T)

    def get_initial_fit(
        self,
        spectra: Sequence[Spectrum],
        continuum_basis: Union[ContinuumBasis, Sequence[ContinuumBasis]] = Sinusoids,
        λ_initial: Optional[float] = 5175,
        R: Optional[float] = None,  
        vsini: Optional[float] = None,
        v_rel: Optional[float] = None,
        op_kwds: Optional[dict] = None,
    ):
        """
        Fit one order to get an initial estimate of the model parameters.
        """

        spectrum = get_closest_order(spectra, λ_initial)
        λ_vacuum, z, inv_sigma_z, oi, pi, S, P = _prepare_spectra([spectrum])
        # TODO: using diferent initial polynomial 
        G, *_ = _continuum_design_matrix(λ_vacuum, Polynomial(2), S, oi)

        basis_vectors, *n_different_bases = self.resample_basis_vectors(λ_vacuum, Ro=R, vsini=vsini, stellar_v_rel=v_rel)

        A = np.hstack([-basis_vectors.T, G])
                
        kwds = dict(
            method="trf", 
            max_iter=10_000, 
            tol=np.finfo(float).eps,
            bounds=self.get_bounds(A.shape[1], *n_different_bases),
            verbose=0,
        )
        kwds.update(op_kwds or {})

        use = (inv_sigma_z > 0)
        use_inv_sigma_z = inv_sigma_z[use]
        A_sigma_z = A[use] * use_inv_sigma_z[:, None]
        Y_sigma_z = z[use] * use_inv_sigma_z

        result = op.lsq_linear(A_sigma_z, Y_sigma_z, **kwds)
        
        n_bases = sum(n_different_bases)

        if v_rel is None:
            continuum = A[:, n_bases:] @ result.x[n_bases:]
            model_rectified_flux = (A @ result.x) / continuum                
            v_shift, *_ = cross_correlate(
                λ_vacuum,
                z/continuum,
                np.ones_like(z),
                λ_vacuum,
                model_rectified_flux,
            )
            v_rel = -v_shift

        fig, ax = plt.subplots()
        ax.plot(λ_vacuum, z, c='k')
        ax.plot(λ_vacuum, A @ result.x, c="tab:red")

        # Compute continuum for all other orders given this rectified flux.
        λ_vacuum, z, inv_sigma_z, oi, pi, S, P = prepared_spectra = _prepare_spectra(spectra)
        G, continuum_basis, n_continuum_params = prepared_continuum = _continuum_design_matrix(λ_vacuum, continuum_basis, S, oi)        
        basis_vectors, *n_different_bases = prepared_basis_vectors = self.resample_basis_vectors(
            λ_vacuum, 
            stellar_v_rel=v_rel,
            Ro=R,
            vsini=vsini
        )

        model_rectified_flux = (-basis_vectors.T) @ result.x[:n_bases]
        use = inv_sigma_z > 0

        p0 = np.zeros(n_bases + G.shape[1])
        p0[:n_bases] = result.x[:n_bases]

        si = 0
        for i, n in enumerate(n_continuum_params):
            mask = (oi == i) * use
            Y = z[mask] - model_rectified_flux[mask]
            Cinv = inv_sigma_z[mask]**2 
            A = G[mask, si:si+n]
            ATCinv = A.T * Cinv
            try:
                p0[n_bases + si:n_bases + si+n] = np.linalg.solve(ATCinv @ A, ATCinv @ Y)
            except:
                None
            si += n
    
        return (p0, v_rel, prepared_spectra, prepared_continuum, prepared_basis_vectors)



    def fit(
        self,
        spectra: Sequence[Spectrum],
        continuum_basis: Union[ContinuumBasis, Sequence[ContinuumBasis]] = Sinusoids,
        R: Optional[float] = None,
        v_rel: Optional[float] = None,
        vsini: Optional[float] = None,
        callback: Optional[callable] = None,
        op_kwds: Optional[dict] = None,
    ):
        """
        Fit the model to some spectra in the observed frame.
        
        :param spectra:
            A sequence of observed spectra.
        
        :param continuum_basis: [optional]
            The continuum basis to use. This can be a single instance of a continuum basis
            or a sequence of continuum bases. If a sequence is given, then the length must
            match the number of orders in the spectra.
        
        :param R: [optional]
            The spectral resolution of the observations. If `None` is given, then no
            convolution will take place; the basis vectors will be interpolated.
        
        :param p0: [optional]
            An initial guess for the model parameters.
        
        :param telluric: [optional]
            Specify whether to model tellurics.
        
        :param callback: [optional]
            A function to call after each iteration of the optimization.
        """
        
        print(f"v_rel_in={v_rel}")
        (p0, v_rel, prepared_spectra, prepared_continuum, prepared_basis_vectors) = self.get_initial_fit(
            spectra, 
            continuum_basis, 
            R=R,
            v_rel=v_rel,
            vsini=vsini,
        )
        print(v_rel)


        λ_vacuum, z, inv_sigma_z, oi, pi, S, P = prepared_spectra                
        G, continuum_basis, n_continuum_params = prepared_continuum        
        basis_vectors, n_stellar_bases, n_telluric_bases = prepared_basis_vectors

        n_bases = n_stellar_bases + n_telluric_bases
        """
        cont = G @ p0[n_bases:]

        in_absorption = (z / cont) < 0.90
        multiplier = np.ones_like(inv_sigma_z)
        multiplier[in_absorption] = 1/100.

        #fig, ax = plt.subplots()
        #ax.plot(λ_vacuum, z / cont)
        
        print("using absorption multiplier")        
        """
        A = np.hstack([-basis_vectors.T, G])
        
        bounds = self.get_bounds(A.shape[1], n_stellar_bases, n_telluric_bases)

        use = (inv_sigma_z > 0)
        use_inv_sigma_z = inv_sigma_z[use] #* multiplier[use]
        A_sigma_z = A[use] * use_inv_sigma_z[:, None]
        Y_sigma_z = z[use] * use_inv_sigma_z

        max_iter, tol = (10_000, np.finfo(float).eps)
        '''
        kwds = dict(
            method="trf", 
            max_iter=1, 
            tol=np.finfo(float).eps,
            bounds=bounds,
            verbose=2,
        )
        kwds.update(op_kwds or {})
        result = op.lsq_linear(A_sigma_z, Y_sigma_z, **kwds)
        '''
        t_init = time()
        result = trf_linear(
            A_sigma_z, 
            Y_sigma_z, 
            p0, 
            *bounds, 
            tol=tol,
            lsq_solver="exact",
            lsmr_tol=tol,
            max_iter=max_iter,
            verbose=2
        )
        t_op = time() - t_init
        
        print(f"took {t_op:.1f} s to optimize {A.shape[1]} parameters with {Y_sigma_z.size} data points")

        n_bases = n_stellar_bases + n_telluric_bases

        if sum(n_continuum_params) > 0:       
            y_pred = [np.exp(A[:, n_bases:] @ result.x[n_bases:])] # continuum
        else:
            y_pred = [np.ones_like(flux)]
        
        y_pred.append(       
            np.exp(A[:, :n_stellar_bases] @ result.x[:n_stellar_bases]) # rectified_stellar_flux
        )
        if n_telluric_bases > 0:
            # rectified_telluric_flux
            y_pred.append(
                np.exp(A[:, n_stellar_bases:n_stellar_bases + n_telluric_bases] @ result.x[n_stellar_bases:n_bases]),
            )
        else:
            y_pred.append(np.nan * np.ones_like(y_pred[0])) # rectified_telluric_flux        
                
        y_pred = np.array(y_pred)
        
        continuum, rectified_model_flux, rectified_telluric_flux = zip(*[y_pred[:, oi == i] for i in range(1 + max(oi))])
        return (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred)



    def get_bounds(self, n_params, n_stellar_bases, n_telluric_bases, stellar_bounds=(0, +np.inf), telluric_bounds=(0, 2)):
        n_bases = n_stellar_bases + n_telluric_bases
        return np.vstack([
            np.tile(stellar_bounds, n_stellar_bases).reshape((n_stellar_bases, 2)),
            np.tile(telluric_bounds, n_telluric_bases).reshape((n_telluric_bases, 2)),
            np.tile([-np.inf, +np.inf], n_params - n_bases).reshape((-1, 2))
        ]).T            


def _continuum_design_matrix(λ_vacuum, continuum_basis, S, oi):
    continuum_basis = _expand_continuum_basis(continuum_basis, S)

    n_continuum_params = [b.num_parameters for b in continuum_basis]
                    
    G = np.zeros((λ_vacuum.size, sum(n_continuum_params)))
    si, order_masks = (0, [])
    for o, (cb, n) in enumerate(zip(continuum_basis, n_continuum_params)):
        mask = (oi == o)
        order_masks.append(mask)
        G[mask, si:si+n] = cb.design_matrix(λ_vacuum[mask])
        si += n    

    return (G, continuum_basis, n_continuum_params)


def get_closest_order(spectra, λ):
    mid = np.array([np.mean(s.λ) for s in spectra])
    diff = np.abs(mid - λ)
    return spectra[np.argmin(diff)]




def _solve_X(flux: Sequence[float], ivar: Sequence[float], A: np.array):
    MTM = A.T @ (ivar[:, None] * A)
    MTy = A.T @ (ivar * flux)
    theta = np.linalg.solve(MTM, MTy)
    return theta
        
        
def instantiate(item, **kwargs):
    if isinstance(item, type):
        return item(**kwargs)
    else:
        return item

def _prepare_spectra(spectra):
    spectra = [spectra] if isinstance(spectra, Spectrum) else spectra
    S = len(spectra)
    P = sum(tuple(map(len, spectra)))
    
    λ_vacuum, z, inv_sigma_z, oi = (np.empty(P), np.empty(P), np.empty(P), np.empty(P, dtype=int))
    for i, (si, spectrum) in enumerate(cumulative(spectra)):
        ei = si + spectrum.λ.size
        oi[si:ei] = i
        λ_vacuum[si:ei] = spectrum.λ_vacuum
        z[si:ei] = np.log(spectrum.flux)
        y = spectrum.flux
        sigma_y = spectrum.ivar**(-0.5)
        sigma_z = sigma_y/y - (sigma_y**2)/(2 * y**2) + (2*sigma_y**3)/(8 * y**3) - (6 * sigma_y**4)/(24 * y**4)
        inv_sigma_z[si:ei] = 1/sigma_z
        
    pi = np.argsort(λ_vacuum) # pixel indices
    λ_vacuum, z, inv_sigma_z, oi = (λ_vacuum[pi], z[pi], inv_sigma_z[pi], oi[pi])
    
    bad_pixel = (~np.isfinite(z)) | (~np.isfinite(inv_sigma_z))
    inv_sigma_z[bad_pixel] = 0
    return (λ_vacuum, z, inv_sigma_z, oi, pi, S, P)


def _expand_continuum_basis(continuum_basis, S):
    if isinstance(continuum_basis, (list, tuple)):
        if len(continuum_basis) != S:
            raise ValueError(f"a sequence of continuum_basis was given, but the length does not match the number of spectra ({len(continuum_basis)} != {S})")
        return tuple(map(instantiate, continuum_basis))
    else:
        return tuple([instantiate(continuum_basis)] * S)


def _interpolate_basis_vector(λ, stellar_vacuum_wavelength, basis_vectors):
    C = basis_vectors.shape[0]
    P = λ.size
    bv = np.zeros((C, P))
    for c, basis_vector in enumerate(basis_vectors):
        bv[c] = np.interp(λ, stellar_vacuum_wavelength, basis_vector, left=0, right=0)
    return bv




if __name__ == "__main__":
    
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np




    spectra = []
    #with fits.open(f"../HARPS.2009-05-12T00-55-28.664_e2ds_A.fits") as image:
    #with fits.open(f"../HARPS.2022-12-08T01-24-25.594_e2ds_A.fits") as image:
    #with fits.open("../ADP.2014-09-23T11-02-38.770/HARPS.2009-05-12T00-55-28.664_e2ds_A.fits") as image:
    #with fits.open("/Users/andycasey/research/continuum/applications/harps/HD109536/HARPS.2009-04-21T00-00-27.670_e2ds_A.fits") as image:
    #with fits.open("/Users/andycasey/research/continuum/applications/harps/alfCenA/archive (5)/HARPS.2012-06-19T23-22-13.907_e2ds_A.fits") as image:
    with fits.open("/Users/andycasey/research/continuum/applications/harps/alfCenA/HARPS.2010-05-18T02-10-08.981_e2ds_A.fits") as image:
        i, coeff = (0, [])
        header_coeff_key = "HIERARCH ESO DRS CAL TH COEFF LL{i}"
        while image[0].header.get(header_coeff_key.format(i=i), False):
            coeff.append(image[0].header[header_coeff_key.format(i=i)])
            i += 1

        n_orders, n_pixels = image[0].data.shape
        x = np.arange(n_pixels)# - n_pixels // 2

        coeff = np.array(coeff).reshape((n_orders, -1))
        λ = np.array([np.polyval(c[::-1], x) for c in coeff])

        n_pixels = 0
        for i in range(n_orders - 1): # ignore last order until we have a good telluric model
            flux = image[0].data[i]
            ivar = 1/image[0].data[i]
            bad_pixel = (flux == 0) | ~np.isfinite(flux) | (flux < 0)
            flux[bad_pixel] = 0
            ivar[bad_pixel] = 0

            na_double = (5898 >= λ[i]) * (λ[i] >= 5886)
            ivar[na_double] = 0

            spectra.append(Spectrum(λ[i], flux, ivar, vacuum=False))
            n_pixels += flux.size


    fig, ax = plt.subplots()
    for i, spectrum in enumerate(spectra):
        ax.plot(spectrum.λ, spectrum.flux, c='k', label=r"$\mathrm{Data}$" if i == 0 else None, zorder=-10)


    #spectra = [spectra[50]]#, spectra[20]]
    #spectra[0].flux = rotational_broadening_kernel(spectra[0].λ, 50, 0.6) @ spectra[0].flux

    #model.stellar_basis_vectors[:, -1] = 0
    #model.stellar_basis_vectors[:, 25] = 0
    with open("../20240816_train_harps_model.pkl", "rb") as fp:
        λ, label_names, parameters, W, H = pickle.load(fp)

    with open("../20240816_train_telfit_model.pkl", "rb") as fp:
        tel_λ, tel_label_names, tel_parameters, tel_W, tel_H = pickle.load(fp)

    # Let's fit every order individually and see what happens.
    model = Clam(
        λ, 
        H,
    )

    (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred) = model.fit(
        spectra, 
        continuum_basis=Polynomial(2),
    )

    import matplotlib
    from mpl_utils import mpl_style
    matplotlib.style.use(mpl_style)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 4), gridspec_kw=dict(width_ratios=(4, 1)))
    
    flux = np.zeros((model.stellar_vacuum_wavelength.size, len(spectra)))
    ivar = np.zeros((model.stellar_vacuum_wavelength.size, len(spectra)))

    combined_flux = np.zeros(model.stellar_vacuum_wavelength.size)

    for i, spectrum in enumerate(spectra):
        for ax in axes[0]:
            ax.plot(spectrum.λ, spectrum.flux, c='k', label=r"$\mathrm{Data}$" if i == 0 else None, zorder=-10)

    model_color = "#1e81b0" # "#4e79a5" # "tab:blue"
    continuum_color = "#063970" # "#77b7b2" # "#666666"
    ylim = axes[0,0].get_ylim()

    for i, spectrum in enumerate(spectra):
        flux[:, i] = np.interp(model.stellar_vacuum_wavelength, spectrum.λ, spectrum.flux / continuum[i], left=0, right=0)
        ivar[:, i] = np.interp(model.stellar_vacuum_wavelength, spectrum.λ, spectrum.ivar * continuum[i]**2, left=0, right=0)
        for ax in (axes[0, 0], axes[0, 1]):
            ax.plot(spectrum.λ, continuum[i], c=continuum_color, label=r"$\mathrm{Continuum~model}$" if i == 0 else None, zorder=3)
            ax.plot(spectrum.λ, continuum[i] * rectified_model_flux[i], c=model_color, label=r"$\mathrm{Stellar~model}$" if i == 0 else None, lw=1, zorder=-4)    
        #axes[0].plot(spectrum.λ, continuum[i] * rectified_telluric_flux[i], c="tab:blue", label=r"$\mathrm{Telluric~model}$" if i == 0 else None, zorder=-5)    
        model_flux = continuum[i] * rectified_model_flux[i] * rectified_telluric_flux[i]

        #axes[0].plot(spectrum.λ, spectrum.flux - continuum[i] * rectified_model_flux[i], c="k")

        #axes[0].plot(spectrum.λ, spectrum.flux - model_flux, c="k")
        #axes[1].plot(spectrum.λ, spectrum.flux / continuum[i], c='k', zorder=-10)
        for ax in (axes[1, 0], axes[1, 1]):
            ax.plot(spectrum.λ, rectified_model_flux[i], c=model_color, zorder=-4)    
        
        #axes[1].plot(spectrum.λ, rectified_telluric_flux[i], c="tab:blue", lw=1, zorder=-5)
        #axes[1].plot(spectrum.λ, spectrum.flux / continuum[i] - (rectified_model_flux[i] * rectified_telluric_flux[i]), c="k")    
    
    non_finite = (~np.isfinite(flux)) | (~np.isfinite(ivar))
    flux[non_finite] = 0
    ivar[non_finite] = 0
    sum_ivar = np.sum(ivar, axis=1)
    combined_flux = np.sum(flux * ivar, axis=1) / sum_ivar

    for ax in (axes[1, 0], axes[1, 1]):
        ax.axhline(1.0, c="#666666", ls=":", lw=0.5)
        ax.plot(model.stellar_vacuum_wavelength, combined_flux, c='k', zorder=-10)

    axes[0, 0].legend(frameon=False, loc="upper left", ncol=4)
    axes[0, 0].set_ylim(ylim)
    for ax in axes[1]:
        ax.set_ylim(-0.1, 1.2)
    axes[1, 0].set_xlabel(r"$\mathrm{Vacuum~wavelength}$ $[\mathrm{\AA}]$")
    axes[0, 0].set_ylabel(r"$\mathrm{Counts}$")
    axes[1, 0].set_ylabel(r"$\mathrm{Rectified~flux}$")
    
    c, e = (3950, 40)
    for ax in (axes[0, 1], axes[1, 1]):
        ax.set_xlim(c - e, c + e)
        ax.set_yticks([])        

    for ax in axes[0]:
        ax.set_xticks([])
    
    
    ptp = np.ptp(axes[0, 0].get_ylim())
    
    for ax in axes[:, 0]:
        ax.set_xlim(spectra[0].λ[0], spectra[-1].λ[-1])

    ylim_max = 2.3e4
    axes[0, 1].set_ylim(-0.05 * ylim_max, ylim_max)
    axes[0, 0].set_ylim(-0.05 * ptp, axes[0,0].get_ylim()[1])
    fig.tight_layout()
    fig.savefig("../../../paper/harps-alfCenA-example.pdf", dpi=300)

