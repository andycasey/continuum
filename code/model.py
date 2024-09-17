
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
from spectrum import Spectrum, SpectrumCollection
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
        
        
    def prepare(self, spectra, vsini=None):

        λ_rest_vacuum, λ_vacuum, *_ = prepared_spectra = _prepare_spectra(spectra)
        
        stellar_basis_vectors = self.stellar_basis_vectors
        if vsini is not None:
            K_lsf = rotational_broadening_kernel(self.stellar_vacuum_wavelength, vsini, 0.6)
            stellar_basis_vectors = stellar_basis_vectors @ K_lsf

        basis_vectors = np.vstack([
            _interpolate_basis_vector(
                λ_rest_vacuum,
                self.stellar_vacuum_wavelength, 
                stellar_basis_vectors
            )
        ])

        if self.can_model_tellurics:
            # need the observed vacuum wavelengths
            telluric_basis_vectors = _interpolate_basis_vector(
                λ_vacuum,
                self.telluric_vacuum_wavelength,
                self.telluric_basis_vectors
            )
            basis_vectors = np.vstack([basis_vectors, telluric_basis_vectors])

        S = self.stellar_basis_vectors.shape[0]
        T = basis_vectors.shape[0] - S
        return (prepared_spectra, basis_vectors, S, T)
    


    def get_initial_fit(
        self,
        spectra: Sequence[Spectrum],
        continuum_basis: Union[ContinuumBasis, Sequence[ContinuumBasis]] = Sinusoids,
        initial_λ: Optional[float] = 5175,
        initial_continuum_basis: ContinuumBasis = Polynomial(1),
        R: Optional[float] = None,  
        vsini: Optional[float] = None,
        op_kwds: Optional[dict] = None,
        initial_order = None,
    ):
        """
        Fit one order to get an initial estimate of the model parameters.
        """
        if initial_order is not None:
            spectrum = initial_order
        else:
            spectrum = get_closest_order(spectra, initial_λ)

        (λ_rest_vacuum, λ_vacuum, z, inv_sigma_z, oi, pi, S, P), basis_vectors, *n_different_bases = self.prepare([spectrum], vsini=vsini)
        
        # TODO: using diferent initial polynomial 
        G, *_ = _continuum_design_matrix(λ_vacuum, initial_continuum_basis, S, oi)

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

        # Compute continuum for all other orders given this rectified flux.
        prepared_spectra, basis_vectors, n_stellar_bases, n_telluric_bases = self.prepare(spectra, vsini=vsini)
        λ_rest_vacuum, λ_vacuum, z, inv_sigma_z, oi, pi, S, P = prepared_spectra
        G, continuum_basis, n_continuum_params = prepared_continuum = _continuum_design_matrix(λ_vacuum, continuum_basis, S, oi)        
        
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
    
        return (p0, prepared_spectra, prepared_continuum, basis_vectors, n_stellar_bases, n_telluric_bases)



    def fit(
        self,
        spectra: Sequence[Spectrum],
        continuum_basis: Union[ContinuumBasis, Sequence[ContinuumBasis]] = Sinusoids,
        vsini: Optional[float] = None,
        op_kwds: Optional[dict] = None,
        **kwargs
    ):
        """
        Fit the model to some spectra in the observed frame.
        
        :param spectra:
            A sequence of observed spectra.
        
        :param continuum_basis: [optional]
            The continuum basis to use. This can be a single instance of a continuum basis
            or a sequence of continuum bases. If a sequence is given, then the length must
            match the number of orders in the spectra.

        :param vsini: [optional]
            The projected rotational velocity of the star in km/s.
        
        :param op_kwds: [optional]
            Keyword arguments to pass to `scipy.optimize._lsq.trf_linear`.
        """
        
        (p0, prepared_spectra, prepared_continuum, basis_vectors, n_stellar_bases, n_telluric_bases) = self.get_initial_fit(
            spectra, 
            continuum_basis, 
            vsini=vsini,
            **kwargs,
        )

        λ_rest_vacuum, λ_vacuum, z, inv_sigma_z, oi, pi, S, P = prepared_spectra                
        G, continuum_basis, n_continuum_params = prepared_continuum        
        A = np.hstack([-basis_vectors.T, G])
        
        bounds = self.get_bounds(A.shape[1], n_stellar_bases, n_telluric_bases)

        use = (inv_sigma_z > 0)
        use_inv_sigma_z = inv_sigma_z[use]
        A_sigma_z = A[use] * use_inv_sigma_z[:, None]
        Y_sigma_z = z[use] * use_inv_sigma_z

        epsilon = np.finfo(float).eps
        kwds = dict(
            x_lsq=p0,
            lb=bounds[0],
            ub=bounds[1],
            tol=epsilon,
            lsmr_tol=epsilon,
            max_iter=10_000,
            lsq_solver="exact",
            verbose=2,
        )
        kwds.update(op_kwds or {})
        t_init = time()
        result = trf_linear(A_sigma_z, Y_sigma_z, **kwds)
        t_op = time() - t_init
        
        print(f"Took {t_op:.1f} s to optimize {A.shape[1]} parameters with {Y_sigma_z.size} data points")

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

        # NaN-ify pixels that were not used in the fit
        y_pred[:, ~use] = np.nan
        
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
    if isinstance(spectra, SpectrumCollection):
        diff = np.abs(np.mean(spectra.λ, axis=1) - λ)
        return spectra.get_order(np.argmin(diff))
    else:    
        mid = np.array([np.mean(s.λ) for s in spectra])
        diff = np.abs(mid - λ)
        return spectra[np.argmin(diff)]




        
        
def instantiate(item, **kwargs):
    if isinstance(item, type):
        return item(**kwargs)
    else:
        return item

def _prepare_spectra(spectra):
    if isinstance(spectra, SpectrumCollection):
        S, P_per_S = spectra.flux.shape
        P = P_per_S * S

        si = 0
        λ_rest_vacuum, λ_vacuum, z, inv_sigma_z, oi = (np.empty(P), np.empty(P), np.empty(P), np.empty(P), np.empty(P, dtype=int))
        for i in range(S):
            ei = si + P_per_S
            oi[si:ei] = i
            λ_vacuum[si:ei] = spectra.λ_vacuum[i]
            λ_rest_vacuum[si:ei] = spectra.λ_rest_vacuum[i]
            z[si:ei] = np.log(spectra.flux[i])
            y = spectra.flux[i]
            sigma_y = spectra.ivar[i]**(-0.5)
            sigma_z = sigma_y/y - (sigma_y**2)/(2 * y**2) + (2*sigma_y**3)/(8 * y**3) - (6 * sigma_y**4)/(24 * y**4)
            inv_sigma_z[si:ei] = 1/sigma_z
            si += P_per_S
    else:
        spectra = [spectra] if isinstance(spectra, Spectrum) else spectra
        S = len(spectra)
        P = sum(tuple(map(len, spectra)))

        λ_rest_vacuum, λ_vacuum, z, inv_sigma_z, oi = (np.empty(P), np.empty(P), np.empty(P), np.empty(P), np.empty(P, dtype=int))
        for i, (si, spectrum) in enumerate(cumulative(spectra)):
            ei = si + spectrum.λ.size
            oi[si:ei] = i
            λ_vacuum[si:ei] = spectrum.λ_vacuum            
            λ_rest_vacuum[si:ei] = spectrum.λ_rest_vacuum
            z[si:ei] = np.log(spectrum.flux)
            y = spectrum.flux
            sigma_y = spectrum.ivar**(-0.5)
            sigma_z = sigma_y/y - (sigma_y**2)/(2 * y**2) + (2*sigma_y**3)/(8 * y**3) - (6 * sigma_y**4)/(24 * y**4)
            inv_sigma_z[si:ei] = 1/sigma_z
        
    pi = np.argsort(λ_vacuum) # pixel indices
    λ_rest_vacuum, λ_vacuum, z, inv_sigma_z, oi = (λ_rest_vacuum[pi], λ_vacuum[pi], z[pi], inv_sigma_z[pi], oi[pi])
    
    bad_pixel = (~np.isfinite(z)) | (~np.isfinite(inv_sigma_z))
    inv_sigma_z[bad_pixel] = 0
    return (λ_rest_vacuum, λ_vacuum, z, inv_sigma_z, oi, pi, S, P)


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
