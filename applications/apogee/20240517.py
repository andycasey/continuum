
import numpy as np
import warnings
import pickle
#from astra.utils import expand_path
from functools import cache
from typing import Optional, Tuple
from scipy import optimize as op
from sklearn.decomposition._nmf import _fit_coordinate_descent
from sklearn.exceptions import ConvergenceWarning
from scipy.signal import fftconvolve

import os 
def expand_path(path):
    return os.path.expanduser(os.path.abspath(path))

@cache
def load_components(path, P, pad=0):    
    with open(expand_path(path), "rb") as fp:
        masked_components = pickle.load(fp)
    if pad > 0:
        components = np.zeros((masked_components.shape[0], P + 2 * pad))
        components[:, pad:-pad] = masked_components
    else:
        components = masked_components
    return components



class Clam(object):

    def __init__(
        self,
        dispersion: np.array,
        components: np.ndarray,
        deg: int,
        regions: Optional[Tuple[Tuple[float, float]]] = None
    ):       
        self.dispersion = dispersion
        _check_dispersion_components_shape(dispersion, components)

        self.regions = regions or [tuple(dispersion[[0, -1]])]
        self.components = components
        self.deg = deg
        self.n_regions = len(self.regions)
        
        A = np.zeros(
            (self.dispersion.size, self.n_regions * self.n_parameters_per_region), 
            dtype=float
        )
        self.region_slices = region_slices(self.dispersion, self.regions)
        for i, region_slice in enumerate(self.region_slices):
            si = i * self.n_parameters_per_region
            ei = (i + 1) * self.n_parameters_per_region
            A[region_slice, si:ei] = design_matrix(dispersion[region_slice], self.deg)

        self.continuum_design_matrix = A
        return None

    @property
    def n_parameters_per_region(self):
        return 2 * self.deg + 1

    def _theta_step(self, flux, ivar, rectified_flux):        
        N, P = flux.shape
        theta = np.zeros((N, self.n_regions, self.n_parameters_per_region))
        continuum = np.nan * np.ones_like(flux)
        continuum_flux = flux / rectified_flux
        continuum_ivar = ivar * rectified_flux**2
        for i in range(N):            
            #for j, (A, mask) in enumerate(zip(*self._dmm)):
            for j, mask in enumerate(self.region_slices):
                sj, ej = (j * self.n_parameters_per_region, (j + 1) * self.n_parameters_per_region)
                A = self.continuum_design_matrix[mask, sj:ej]
                MTM = A.T @ (continuum_ivar[i, mask][:, None] * A)
                MTy = A.T @ (continuum_ivar[i, mask] * continuum_flux[i, mask])
                try:
                    theta[i, j] = np.linalg.solve(MTM, MTy)
                except np.linalg.LinAlgError:
                    if np.any(continuum_ivar[i, mask] > 0):
                        raise
                continuum[i, mask] = A @ theta[i, j]        

        return (theta, continuum)


    def _W_step(self, mean_rectified_flux, W, **kwargs):
        absorption = 1 - mean_rectified_flux
        use = np.zeros(mean_rectified_flux.size, dtype=bool)
        for region_slice in self.region_slices:
            use[region_slice] = True
        use *= (
            np.isfinite(absorption) 
        &   (absorption >= 0) 
        &   (mean_rectified_flux > 0)
        )
        W_next, _, n_iter = _fit_coordinate_descent(
            absorption[use].reshape((1, -1)),
            W,
            self.components[:, use],
            update_H=False,
            verbose=False,
            shuffle=True
        )        
        rectified_model_flux = 1 - (W_next @ self.components)[0]
        return (W_next, rectified_model_flux, np.sum(use), n_iter)


    def get_initial_guess_by_iteration(self, flux, ivar, A=None, max_iter=32):
        C, P = self.components.shape
        ivar_sum = np.sum(ivar, axis=0)
        no_data = ivar_sum == 0
        rectified_flux = np.ones(P)
        continuum = np.ones_like(flux)
        W = np.zeros((1, C), dtype=np.float64)

        thetas, chi_sqs = ([], [])
        with warnings.catch_warnings():
            for category in (RuntimeWarning, ConvergenceWarning):
                warnings.filterwarnings("ignore", category=category)

            for iteration in range(max_iter):
                theta, continuum = self._theta_step(
                    flux,
                    ivar,
                    rectified_flux,
                )                
                mean_rectified_flux = np.sum((flux / continuum) * ivar, axis=0) / ivar_sum
                mean_rectified_flux[no_data] = 0.0
                W, rectified_flux, n_pixels_used, n_iter = self._W_step(mean_rectified_flux, W)
                chi_sqs.append(np.nansum((flux - rectified_flux * continuum)**2 * ivar))
                if iteration > 0 and (chi_sqs[-1] > chi_sqs[-2]):
                    break            

                thetas.append(np.hstack([W.flatten(), theta.flatten()]))

        return thetas[-1]


    def continuum(self, wavelength, theta):
        C, P = self.components.shape
        
        A = np.zeros(
            (wavelength.size, self.n_regions * self.n_parameters_per_region), 
            dtype=float
        )
        for i, region_slice in enumerate(self.region_slices):
            si = i * self.n_parameters_per_region
            ei = (i + 1) * self.n_parameters_per_region
            A[region_slice, si:ei] = design_matrix(wavelength[region_slice], self.deg)
        
        return (A @ theta).reshape((-1, P))

    def _predict(self, theta, A_slice, C, P):
        return (1 - theta[:C] @ self.components) * (A_slice @ theta[C:]).reshape((-1, P))

            
    
    def __call__(self, theta, A_slice=None, full_output=False):
        C, P = self.components.shape
        
        if A_slice is None:
            T = len(theta)
            R = self.n_regions
            N = int((T - C) / (R * (self.n_parameters_per_region)))
            A = self.full_design_matrix(N)
            A_slice = A[:, C:]

        rectified_flux = 1 - theta[:C] @ self.components
        continuum = (A_slice @ theta[C:]).reshape((-1, P))
        flux = rectified_flux * continuum

        if not full_output:
            return flux
        return (flux, rectified_flux, continuum)
    

    def full_design_matrix(self, N):        
        C, P = self.components.shape
        R = len(self.regions)

        K = R * self.n_parameters_per_region
        A = np.zeros((N * P, C + N * K), dtype=float)
        for i in range(N):
            A[i*P:(i+1)*P, :C] = self.components.T
            A[i*P:(i+1)*P, C + i*K:C + (i+1)*K] = self.continuum_design_matrix
        return A

    def get_mask(self, ivar):
        N, P = np.atleast_2d(ivar).shape        
        use = np.zeros((N, P), dtype=bool)
        for region_slice in self.region_slices:
            use[:, region_slice] = True
        use *= (ivar > 0)
        return ~use

    

    def get_initial_guess_with_small_W(self, flux, ivar, A=None, small=1e-12):
        with warnings.catch_warnings():        
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            N, P = flux.shape
            if A is None:
                A = self.full_design_matrix(N)
            Y = flux.flatten()
            use = ~self.get_mask(ivar).flatten()
            result = op.lsq_linear(
                A[use],
                Y[use],
                bounds=self.get_bounds(N, [-np.inf, 0]),
            )
            
            C, P = self.components.shape
            return np.hstack([small * np.ones(C), result.x[C:]])
                    

    def get_initial_guess_by_linear_least_squares_with_bounds(self, flux, ivar, A=None):
        print("USING LLSQB for first guess")
        with warnings.catch_warnings():        
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            N, P = flux.shape
            if A is None:
                A = self.full_design_matrix(N)
            Y = flux.flatten()
            use = ~self.get_mask(ivar).flatten()            
            result = op.lsq_linear(
                A[use],
                Y[use],
                bounds=self.get_bounds(N, [-np.inf, 0]),
            )
            C, P = self.components.shape
            continuum = (A[:, C:] @ result.x[C:]).reshape(flux.shape)

            mean_rectified_flux, _ = self.get_mean_rectified_flux(flux, ivar, continuum)
            W_next, *_ = self._W_step(mean_rectified_flux, result.x[:C].astype(np.float64).reshape((1, C))) #np.zeros((1, C), dtype=np.float64))        
            return np.hstack([W_next.flatten(), result.x[C:]])


    def get_mean_rectified_flux(self, flux, ivar, continuum):
        """
        Compute a mean rectified spectrum, given an estimate of the contiuum.
        
        :param flux:
            A (N, P) shape array of flux values.
        
        :param ivar:
            A (N, P) shape array of inverse variances on flux values.
        
        :param continuum: 
            A (N, P) shape array of estimated continuum fluxes.
        """
        N, P = flux.shape
        if N == 1:
            return ((flux / continuum)[0], (ivar * continuum**2)[0])
        ivar_sum = np.sum(ivar, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            mean_rectified_flux = np.sum((flux / continuum) * ivar, axis=0) / ivar_sum
        no_data = ivar_sum == 0
        mean_rectified_flux[no_data] = 0.0
        mean_rectified_ivar = np.mean(ivar * continuum**2, axis=0)        
        return (mean_rectified_flux, mean_rectified_ivar)


    def fit_stellar_parameters(
        self,
        flux,
        ivar,
        basis_weight_interpolator,
        random_forest_regressor,
        label_names,
        unscale,
        full_output=True,
        fit_vsini=False,
        x0_params=None,
        x0_vsini=100,
        **kwargs
    ):
        try:
            self.A1
        except:
            self.A1 = self.full_design_matrix(1)
            
        if x0_params is None:
            _, meta_init = self.fit(flux, ivar, full_output=True)

            x0_params, = random_forest_regressor.predict(meta_init["W"].reshape((1, -1)))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            if kwargs.get("check_shapes", True):
                flux, ivar = _check_and_reshape_flux_ivar(self.dispersion, flux, ivar)

            N, P = flux.shape
            A = self.A1 # self.full_design_matrix(N)

            try:
                L = len(basis_weight_interpolator.grid) # number of stellar parameters
            except:
                L = len(x0_params)
            
            use = ~self.get_mask(ivar).flatten()
            sigma = ivar**-0.5
            
            C, P = self.components.shape
            A_slice = A[:, C:]
            
            # given x0_params, predict initial W coefficients.
            W_init = np.clip(basis_weight_interpolator(x0_params), 0, np.inf)
            init_rectified_model_flux = (1 - W_init @ self.components).flatten()
            
            '''
            x02 = self.get_initial_guess_with_small_W(
                flux / init_rectified_model_flux, 
                ivar * init_rectified_model_flux**2,
                A, 
                small=1e-10
            )
            '''
            # TODO: put to function
            use = ~self.get_mask(ivar * init_rectified_model_flux**2).flatten()
            Y = (flux.flatten() / init_rectified_model_flux)[use]
            C_inv = np.diag((ivar[0] * init_rectified_model_flux**2)[use])
            
            B = A[use, C:]
            x0_continuum = np.linalg.solve(B.T @ C_inv @ B, B.T @ C_inv @ Y)
                                                
            x0 = np.hstack([
                x0_params, # stellar params  
                x0_continuum # continuum params,
            ])
            
            if fit_vsini:
                x0 = np.hstack([x0, x0_vsini])
                            
            def f(_, *params, full_output=False):
                W = np.clip(basis_weight_interpolator(params[:L]), 0, np.inf)
                rectified_model_flux = (1 - W @ self.components).flatten()
                continuum = (A_slice @ params[L:L+A_slice.shape[1]]).flatten()
                
                if fit_vsini:
                    rectified_model_flux = apply_rotation(rectified_model_flux, params[-1])
                    
                if full_output:
                    continuum[~use] = np.nan
                    model_flux = rectified_model_flux * continuum                
                    return (W, model_flux, rectified_model_flux, continuum)
                
                model_flux = rectified_model_flux * continuum                
                return model_flux[use]
                                    
            def chi2(params):
                return np.nansum((f(None, *params) - flux.flatten()[use])**2 * ivar.flatten()[use])
            
            
            bound_entries = [
                np.tile([0, 1], L).reshape((L, 2)),
                np.tile([-np.inf, +np.inf], self.n_regions * self.n_parameters_per_region).reshape((-1, 2)),                
            ]
            if fit_vsini:
                bound_entries.append([0, 100])
                
            bounds = np.vstack(bound_entries).T
            
            p_opt, cov = op.curve_fit(
                f,
                None,
                flux.flatten()[use],
                p0=x0,
                sigma=sigma.flatten()[use],
                bounds=bounds
            )
            
            W, model_flux, rectified_model_flux, continuum = f(None, *p_opt, full_output=True)
            continuum[~use] = np.nan
            
            chi2 = ((model_flux - flux)**2 * ivar).flatten()
            chi2[~use] = np.nan
            rchi2 = np.sum(chi2[use]) / (use.sum() - p_opt.size - 1)

            # the first N_component values of p_opt are the W coefficients
            # the remaining values are the theta coefficients
            meta = dict(
                cov=cov,
                theta=p_opt,
                absoprtion_coefficients=W,
                continuum_coefficients=p_opt[-A_slice.shape[1]:],
                model_flux=model_flux,
                rectified_model_flux=rectified_model_flux,
                continuum=continuum,
                mask=~use,
                pixel_chi2=chi2,
                rchi2=rchi2
            )

    
        
        if full_output:
            L = len(label_names)        
            meta["stellar_parameters"] = dict(zip(label_names,  unscale(meta["theta"][:L])))
            meta["initial_stellar_parameters"] = dict(zip(label_names, unscale(x0_params)))
            
            # TODO: use cov to estimate uncertainties on stellar parameters
            
            return (continuum, meta)
        return continuum
        

    def fit_with_stellar_parameters(
        self,
        flux,
        ivar,
        basis_weight_interpolator,
        fit_vsini=False,
        x0=None,   
        full_output=True,  
        x0_params=[0.5, 0.5, 0.5, 0.5],
        x0_vsini=5,
        **kwargs
    ):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            if kwargs.get("check_shapes", True):
                flux, ivar = _check_and_reshape_flux_ivar(self.dispersion, flux, ivar)

            N, P = flux.shape
            A = self.full_design_matrix(N)

            try:
                L = len(basis_weight_interpolator.grid) # number of stellar parameters
            except:
                L = len(x0_params)
            
            use = ~self.get_mask(ivar).flatten()
            sigma = ivar**-0.5
            
            C, P = self.components.shape
            A_slice = A[:, C:]
            
            x0_full = self.get_initial_guess_with_small_W(flux, ivar, A, small=1e-10)
            
            N * self.n_regions * (self.n_parameters_per_region)
            
            x0 = list(np.hstack([x0_params, x0_full[C:]]))
            
            if fit_vsini:
                x0.append(x0_vsini)
                
            
            def f(_, *params, full_output=False):
                W = np.clip(basis_weight_interpolator(params[:L]), 0, np.inf)
                rectified_model_flux = (1 - W @ self.components).flatten()
                continuum = (A_slice @ params[L:L+A_slice.shape[1]]).flatten()
                
                if fit_vsini:
                    rectified_model_flux = apply_rotation(rectified_model_flux, params[-1])
                    
                if full_output:
                    continuum[~use] = np.nan
                    model_flux = rectified_model_flux * continuum                
                    return (W, model_flux, rectified_model_flux, continuum)
                
                model_flux = rectified_model_flux * continuum                
                return model_flux[use]
            
            def chi2(params):
                return np.nansum((f(None, *params) - flux.flatten()[use])**2 * ivar.flatten()[use])
            
            
            bound_entries = [
                np.tile([0, 1], L).reshape((L, 2)),
                np.tile([-np.inf, +np.inf], self.n_regions * self.n_parameters_per_region).reshape((-1, 2)),                
            ]
            if fit_vsini:
                bound_entries.append([0, 100])
                
            bounds = np.vstack(bound_entries).T
            
            p_opt, cov = op.curve_fit(
                f,
                None,
                flux.flatten()[use],
                p0=x0,
                sigma=sigma.flatten()[use],
                bounds=bounds
            )
            
            W, model_flux, rectified_model_flux, continuum = f(None, *p_opt, full_output=True)
            continuum[~use] = np.nan
            
            chi2 = ((model_flux - flux)**2 * ivar).flatten()
            chi2[~use] = np.nan
            rchi2 = np.sum(chi2[use]) / (use.sum() - p_opt.size - 1)

            # the first N_component values of p_opt are the W coefficients
            # the remaining values are the theta coefficients
            result = dict(
                cov=cov,
                theta=p_opt,
                absoprtion_coefficients=W,
                continuum_coefficients=p_opt[-A_slice.shape[1]:],
                model_flux=model_flux,
                rectified_model_flux=rectified_model_flux,
                continuum=continuum,
                mask=~use,
                pixel_chi2=chi2,
                rchi2=rchi2
            )

            if full_output:
                return (continuum, result)
            else:
                return continuum

    def fit_with_vsini(self, flux: np.ndarray, ivar: np.ndarray, x0=None, vsini_epsilon=1e-2, op_kwds=dict(xtol=1e-16, verbose=0), **kwargs):
        if kwargs.get("check_shapes", True):
            flux, ivar = _check_and_reshape_flux_ivar(self.dispersion, flux, ivar)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            N, P = flux.shape
            A = self.full_design_matrix(N)
            if x0 is None:
                x0 = list(self.get_initial_guess_with_small_W(flux, ivar, A)) + [kwargs.get("x0_vsini", 5)]

            use = ~self.get_mask(ivar).flatten()
            
            C, P = self.components.shape
            A_slice = A[:, C:]
            
            flux = flux.flatten()
            ivar = ivar.flatten() # TODO: decide how we want to handle these 
            # Pre-compute some things for faster Jacobian evaluations
            inv_sigma = np.sqrt(ivar)
            mBVT = -self.components.T
            A_inv_sigma = A_slice * inv_sigma[:, None]
            mBVT_inv_sigma = mBVT * inv_sigma[:, None]
            
            # function to approximate dy/dvsini
            def _apply_rotation(vsini, *args):
                rectified_flux, continuum = args
                return (apply_rotation(rectified_flux, np.atleast_1d(vsini)[0]) * continuum - flux) * inv_sigma
                            
            def jacobian(theta):
                # compute rectified flux and continuum
                rectified_flux = 1 - theta[:C] @ self.components
                continuum = A_slice @ theta[C:-1]
                
                return np.hstack([
                    (A_slice @ theta[C:-1, None]) * mBVT_inv_sigma, # dy/dW (basis weights)
                    (1 + mBVT @ theta[:C, None]) * A_inv_sigma, # dy/dphi (continuum)
                    op.approx_fprime(theta[-1], _apply_rotation, vsini_epsilon, rectified_flux, continuum)
                ])                

            def f(theta, full_output=False):
                rectified_flux = (1 - theta[:C] @ self.components)                
                convolved_flux = apply_rotation(rectified_flux, theta[-1])
                continuum = (A_slice @ theta[C:-1])
                y = convolved_flux * continuum 
                if not full_output:
                    return (y - flux) * inv_sigma
                return (rectified_flux, convolved_flux, continuum)
            
            
            bounds = np.vstack([
                np.tile([0, np.inf], C).reshape((C, 2)),
                np.tile([-np.inf, +np.inf], A_slice.shape[1]).reshape((-1, 2)),
                [0, 100]
            ]).T
            
            result = op.least_squares(
                f, 
                x0, 
                jac=jacobian, 
                bounds=bounds,
                **op_kwds
            )
            
            rectified_flux, convolved_flux, continuum = f(result.x, True)
            model_flux = convolved_flux * continuum
            
            vsini = result.x[-1]
            meta = dict(
                result=result,
                model_flux=model_flux,
                convolved_flux=convolved_flux,
                rectified_flux=rectified_flux,
                continuum=continuum,
                W=result.x[:-1],
                vsini=result.x[-1]
            )
            return (vsini, meta)
    
                        

    def fit(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        x0=None,
        full_output=False,
        **kwargs
    ):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            if kwargs.get("check_shapes", True):
                flux, ivar = _check_and_reshape_flux_ivar(self.dispersion, flux, ivar)

            N, P = flux.shape
            A = self.full_design_matrix(N)
            if x0 is None:
                x0_callables = (
                    self.get_initial_guess_with_small_W, # this one is faster and seems to be less prone to getting into runtimeerrors at optimisation time
                    self.get_initial_guess_by_linear_least_squares_with_bounds,
                )
            else:
                x0_callables = [lambda *_: x0]
        
            # build a mas using ivar and region masks
            use = ~self.get_mask(ivar).flatten()
            sigma = ivar**-0.5
            
            C, P = self.components.shape
            A_slice = A[:, C:]
            
            def f(_, *params):
                
                return self._predict(params, A_slice=A_slice, C=C, P=P).flatten()[use]
                #chi2 = (r - flux.flatten()[use]) * ivar.flatten()[use]
                #print(np.nansum(chi2))#, *params)
                #return r
                
            
            for x0_callable in x0_callables:
                x0 = x0_callable(flux, ivar, A)
                try:                            
                    p_opt, cov = op.curve_fit(
                        f,
                        None,
                        flux.flatten()[use],
                        p0=x0,
                        sigma=sigma.flatten()[use],
                        bounds=self.get_bounds(flux.shape[0])
                    )
                except RuntimeError:
                    continue
                else:
                    break
            else:
                raise RuntimeError(f"Optimization failed")
            
            model_flux, rectified_model_flux, continuum = self(p_opt, full_output=True)

            chi2 = ((model_flux - flux)**2 * ivar).flatten()
            chi2[~use] = np.nan
            rchi2 = np.sum(chi2[use]) / (use.sum() - p_opt.size - 1)

            # the first N_component values of p_opt are the W coefficients
            # the remaining values are the theta coefficients
            result = dict(
                W=p_opt[:self.components.shape[0]],
                theta=p_opt[self.components.shape[0]:],
                model_flux=model_flux,
                rectified_model_flux=rectified_model_flux,
                continuum=continuum,
                mask=~use,
                pixel_chi2=chi2,
                rchi2=rchi2
            )

            if full_output:
                return (continuum, result)
            else:
                return continuum


    def get_bounds(self, N, component_bounds=(0, +np.inf)):
        C, P = self.components.shape          
        A = N * self.n_regions * (self.n_parameters_per_region)

        return np.vstack([
            np.tile(component_bounds, C).reshape((C, 2)),
            np.tile([-np.inf, +np.inf], A).reshape((-1, 2))
        ]).T            



def region_slices(dispersion, regions):
    slices = []
    for region in regions:
        si, ei = dispersion.searchsorted(region)
        slices.append(slice(si, ei + 1))
    return slices


def _check_and_reshape_flux_ivar(dispersion, flux, ivar):
    P = dispersion.size
    flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
    N1, P1 = flux.shape
    N2, P2 = ivar.shape

    assert (N1 == N2) and (P1 == P2), "`flux` and `ivar` do not have the same shape"
    assert (P == P1), f"Number of pixels in flux does not match dispersion array ({P} != {P1})"

    bad_pixel = (
        (~np.isfinite(flux))
    |   (~np.isfinite(ivar))
    |   (flux <= 0)
    )
    flux[bad_pixel] = 0
    ivar[bad_pixel] = 0
    return (flux, ivar)


def _check_dispersion_components_shape(dispersion, components):
    P = dispersion.size
    assert dispersion.ndim == 1, "Dispersion must be a one-dimensional array." 
    C, P2 = components.shape
    assert P == P2, "`components` should have shape (C, P) where P is the size of `dispersion`"


def design_matrix(dispersion: np.array, deg: int) -> np.array:
    #L = 1300.0
    #scale = 2 * (np.pi / (2 * np.ptp(dispersion)))
    scale = np.pi / np.ptp(dispersion)
    return np.vstack(
        [
            np.ones_like(dispersion).reshape((1, -1)),
            np.array(
                [
                    [np.cos(o * scale * dispersion), np.sin(o * scale * dispersion)]
                    for o in range(1, deg + 1)
                ]
            ).reshape((2 * deg, dispersion.size)),
        ]
    ).T


c_cgs = 2.99792458e10  # speed of light in cm/s, adjust this value as per your requirements

def apply_rotation(flux, vsini, epsilon=0.6, log_lambda_step=1.3815510557964276e-5):
    if vsini == 0:
        return flux

    # calculate log-lamgda step from the wavelength grid
    # log_lambda_step = mean(diff(log.(ferre_wls)))

    # half-width of the rotation kernel in Δlnλ
    delta_ln_lambda_max = vsini * 1e5 / c_cgs
    # half-width of the rotation kernel in pixels
    p = delta_ln_lambda_max / log_lambda_step
    # Δlnλ detuning for each pixel in kernel
    delta_ln_lambdas = np.concatenate(([-delta_ln_lambda_max], np.arange(-np.floor(p), np.floor(p)+1)*log_lambda_step, [delta_ln_lambda_max]))
    if len(delta_ln_lambdas) == 2:
        return flux

    # kernel coefficients
    c1 = 2*(1-epsilon)
    c2 = np.pi * epsilon / 2
    c3 = np.pi * (1-epsilon/3)

    x = 1 - (delta_ln_lambdas/delta_ln_lambda_max)**2
    rotation_kernel = (c1*np.sqrt(x) + c2*x) / (c3 * delta_ln_lambda_max)

    rotation_kernel[rotation_kernel == 0] = 0

    rotation_kernel /= np.sum(rotation_kernel)
    #print("r ", len(rotation_kernel))
    offset = (len(rotation_kernel) - 1) // 2

    return fftconvolve(flux, rotation_kernel, mode='full')[offset : -offset]


if __name__ == "__main__":

    import os
    import h5py as h5
    import numpy as np
    import pickle
    from sklearn.decomposition import NMF
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.interpolate import RegularGridInterpolator
    from sklearn.ensemble import RandomForestRegressor
    #from astra.models import Source
    #from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
    #from astra.models.mwm import ApogeeCombinedSpectrum
    from scipy import optimize as op
    from tqdm import tqdm
    from astropy.table import Table    

    
    NMF_PATH = "20240517_components.pkl"
    SLICE_ON_N_ONLY = False
    SLICE_ON_C_AND_N = True

    try:
        label_names
    except NameError:
        with h5.File("big_grid_2024-12-20.h5", "r") as fp:
            label_names = ["Teff", "logg", "vmic", "m_h", "c_m", "n_m"][::-1]
            grid_points = [fp[f"{ln}_vals"][:] for ln in label_names]
            grid_model_flux = fp["spectra"][:]

        print(grid_model_flux.shape)
        print(tuple(map(len, grid_points)))

        if sum([SLICE_ON_N_ONLY, SLICE_ON_C_AND_N]) > 1:
            raise ValueError("Only one of SLICE_ON_N_ONLY or SLICE_ON_C_AND_N can be True")

        if SLICE_ON_N_ONLY:
            # Take a slice
            n_m = 0
            n_m_index = list(grid_points[label_names.index("n_m")]).index(n_m)

            label_names = label_names[1:]
            grid_points = grid_points[1:]
            grid_model_flux = grid_model_flux[n_m_index]
                
        
        elif SLICE_ON_C_AND_N:
                
            # Take a slice
            c_m = n_m = 0
            c_m_index = list(grid_points[label_names.index("c_m")]).index(c_m)
            n_m_index = list(grid_points[label_names.index("n_m")]).index(n_m)

            label_names = label_names[2:]
            grid_points = grid_points[2:]
            grid_model_flux = grid_model_flux[n_m_index, c_m_index]
        
        
        n_pixels = grid_model_flux.shape[-1]
        
        absorption = np.clip(1 - grid_model_flux, 0, 1).reshape((-1, n_pixels))
        absorption[~np.isfinite(absorption)] = 0
        
    else:
        print("Using existing grid")


    if os.path.exists(NMF_PATH):        
        with open(NMF_PATH, "rb") as fp:
            W, H, meta = pickle.load(fp)
        print(f"Loaded from {NMF_PATH}: {meta}")
        
    else:
        # Train the NMF model

        kwds = dict(
            n_components=16,
            solver="cd",
            max_iter=1_000,
            tol=1e-4,
            random_state=0,
            verbose=1,
            alpha_H=0.0,
            l1_ratio=0.0,            
        )
        
        nmf_model = NMF(**kwds)

        W = nmf_model.fit_transform(absorption)
        H = nmf_model.components_

        meta = kwds.copy()

        with open(NMF_PATH, "wb") as fp:
            pickle.dump((W, H, meta), fp)

    
    clam_model = Clam(
        dispersion=10**(4.179 + 6e-6 * np.arange(8575)),
        components=H,
        deg=3,
        regions=[
            [15161.84316643 - 35, 15757.66995776 + 60],
            [15877.64179911 - 25, 16380.98452330 + 60],
            [16494.30420468 - 30, 16898.18264895 + 60]
        ]
    )
    
    wavelength = 10**(4.179 + 6e-6 * np.arange(8575))
    grid = np.array([[gp[j] for gp, j in zip(grid_points, np.unravel_index(i, grid_model_flux.shape[:-1]))] for i in range(absorption.shape[0])])    
    weighted_sums = np.array([np.sum(W[:, [i]] * grid / np.sum(W[:, i]), axis=0) for i in range(H.shape[0])])

    indices = np.argsort(weighted_sums[:, -1])
            
    min_grid_points = np.array(list(map(np.min, grid_points)))
    ptp_grid_points = np.array(list(map(np.ptp, grid_points)))

    def scale(grid_points):
        return (grid_points - min_grid_points) / ptp_grid_points

    def unscale(norm_grid_points):
        return norm_grid_points * ptp_grid_points + min_grid_points

    norm_grid_points = [(g - min_grid_points[i]) / ptp_grid_points[i] for i, g in enumerate(grid_points)]

    interpolator = RegularGridInterpolator(
        tuple(norm_grid_points),
        W.reshape((*grid_model_flux.shape[:-1], -1)),
        method="linear",
        bounds_error=False,
        fill_value=0
    )

    # train a random forest regressor    
    rf_model = RandomForestRegressor(
        random_state=0,
        n_estimators=10,
        max_depth=None,
        n_jobs=None, # Note: setting this to -1 makes things way slower for inference time because the pool bottlenecks
    )
    rf_model.fit(W, scale(grid))
    
    from time import time
    t_init = time()
    pred_param = unscale(rf_model.predict(W))
    print(f"prediction time: {time() - t_init:.2f}s")
    diff = pred_param - grid
    
    print(f"Label bias / standard deviation")
    for i, label_name in enumerate(label_names):
        print(f"{label_name} {np.mean(diff[:, i]):.2f} {np.std(diff[:, i]):.2f}")

    import pickle
    with open("20240517_spectra.pkl", "rb") as fp:
        flux, ivar, all_meta = pickle.load(fp)
        
    for index, item in enumerate(all_meta):
        if item["spectrum_pk"] == 16158599:
            break
        #if item["spectrum_pk"] == 16439299:
        #    break
    

    
    t_init = time()
    _, meta = clam_model.fit_stellar_parameters(
        flux[index],
        ivar[index],
        basis_weight_interpolator=interpolator,
        random_forest_regressor=rf_model,
        label_names=label_names,
        unscale=unscale,
        full_output=True,
    )
    print(time() - t_init)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(wavelength, flux[index], c='k')
    ax.plot(wavelength, meta["model_flux"], c="tab:red")
    print(meta["initial_stellar_parameters"])
    print(meta["stellar_parameters"])
    print(meta["rchi2"])
    
    
    IMAGE_SUFFIX = "cm_0_init_6000_4"
    # Do M67
    N, P = flux.shape
    
    done = []
    m67_results = []
    
    for i in tqdm(range(N)):
        if all_meta[i]["sdss4_apogee_member_flags"] != 2**18: # M67
            continue
        
        if all_meta[i]["spectrum_pk"] in done:
            continue
        
        try:
            _, meta = clam_model.fit_stellar_parameters(
                flux[i],
                ivar[i],
                basis_weight_interpolator=interpolator,
                random_forest_regressor=rf_model,
                label_names=label_names,
                unscale=unscale,
                full_output=True,
                #fit_vsini=True,
                x0_params=scale([0, 1, 4, 6000])
            )
        except KeyboardInterrupt:
            break
        except:
            continue
        else:
            done.append(all_meta[i]["spectrum_pk"])
            m67_results.append((i, _, meta))
        
        
    X = np.array([list(r[2]["stellar_parameters"].values()) + [r[2]["theta"][-1]] for r in m67_results])
    fig, ax = plt.subplots()
    scat = ax.scatter(
        X[:, label_names.index("Teff")],
        X[:, label_names.index("logg")],
        c=X[:, label_names.index("m_h")],
        #vmin=0,
        #vmax=10,
        s=5
    )
    ax.set_xlim(*grid_points[label_names.index("Teff")][[-1, 0]])
    ax.set_ylim(*grid_points[label_names.index("logg")][[-1, 0]])
    ax.set_xlabel("teff")
    ax.set_ylabel("logg")
    cbar = plt.colorbar(scat)
    cbar.set_label("m_h")
    ax.set_title("M67")
    fig.tight_layout()
    fig.savefig(f"20240517_m67_{IMAGE_SUFFIX}.png", dpi=300)
    
    ngc188_results = []
    for i in tqdm(range(N)):
        if all_meta[i]["sdss4_apogee_member_flags"] != 2**17: # NGC 188
            continue
        
        if all_meta[i]["spectrum_pk"] in done:
            continue
        
        try:
            _, meta = clam_model.fit_stellar_parameters(
                flux[i],
                ivar[i],
                basis_weight_interpolator=interpolator,
                random_forest_regressor=rf_model,
                label_names=label_names,
                unscale=unscale,
                full_output=True,
            )
        except KeyboardInterrupt:
            break
        except:
            continue
        else:
            done.append(all_meta[i]["spectrum_pk"])
            ngc188_results.append((i, _, meta))    
    

    
    X = np.array([list(r[2]["stellar_parameters"].values()) for r in ngc188_results])
    fig, ax = plt.subplots()
    scat = ax.scatter(
        X[:, label_names.index("Teff")],
        X[:, label_names.index("logg")],
        c=X[:, label_names.index("m_h")],
    )
    ax.set_xlim(7000, 3000)
    ax.set_ylim(5.5, 0)
    ax.set_xlabel("teff")
    ax.set_ylabel("logg")
    cbar = plt.colorbar(scat)
    cbar.set_label("m_h")
    ax.set_title("NGC188")
    fig.tight_layout()
    fig.savefig(f"20240517_ngc188_{IMAGE_SUFFIX}.png", dpi=300)
        
    raise a
        
    sun = ApogeeCombinedSpectrum.get(source_pk=Source.get(sdss4_apogee_id="VESTA").pk)
    
    _, meta = clam_model.fit_stellar_parameters(
        sun.flux,
        sun.ivar,
        basis_weight_interpolator=interpolator,
        random_forest_regressor=rf_model,
        label_names=label_names,
        unscale=unscale,
        full_output=True
    )
    
    
    # do some cluster stars now.
    
    ngc188_spectra = list(
        ApogeeCombinedSpectrum
        .select()
        .join(Source)
        .where(Source.flag_sdss4_apogee_member_ngc_188)    
    )

    rows = []
    for spectrum in tqdm(ngc188_spectra):
        
        _, meta = clam_model.fit_stellar_parameters(
            spectrum.flux,
            spectrum.ivar,
            basis_weight_interpolator=interpolator,
            random_forest_regressor=rf_model,
            label_names=label_names,
            unscale=unscale,
            full_output=True
        )               
        rows.append(meta["stellar_parameters"])
    
    rows = Table(rows=rows)

    from astra.models import ASPCAP
    
    q = (
        ASPCAP
        .select()
        .join(ApogeeCoaddedSpectrumInApStar, on=(ASPCAP.spectrum_pk == ApogeeCoaddedSpectrumInApStar.spectrum_pk))
        .join(Source, on=(ApogeeCoaddedSpectrumInApStar.source_pk == Source.pk))
        .where(
            Source.flag_sdss4_apogee_member_ngc_188
        )
    )
    aspcap_results = np.array([(r.m_h_atm, r.logg, r.teff) for r in q])
    
    fig, ax = plt.subplots()
    ax.scatter(
        aspcap_results[:, 2], # teff
        aspcap_results[:, 1], # logg
        c=aspcap_results[:, 0], # m_h
        marker='s',
        alpha=0.1,
        zorder=-1,
        label="ASPCAP (calibrated)"
        
    )
    
    scat = ax.scatter(
        rows["Teff"],
        rows["logg"],
        c=rows["m_h"],
        label="CLAM (raw)"
    )
    cbar = plt.colorbar(scat)    
    ax.set_xlim(7000, 3000)
    ax.set_ylim(5.5, 0)
    ax.set_xlabel("teff")
    ax.set_ylabel("logg")
    ax.legend()
    fig.savefig("/uufs/chpc.utah.edu/common/home/u6020307/20240517_ngc188.png")
    
    print("NGC 188 done")
    
    # Now let's do all the star clusters.
    
    spectra = list(
        ApogeeCombinedSpectrum
        .select(
            ApogeeCombinedSpectrum,
            Source.sdss4_apogee_member_flags
        )
        .join(Source)
        .where(Source.sdss4_apogee_member_flags > 0)    
        .objects()
    )    
    
    print("Should cache this because it's annoying to load each time")
    
    spectrum_meta, fluxs, ivars = ([], [], [])
    for spectrum in tqdm(spectra):
        try:
            f, i = (spectrum.flux, spectrum.ivar)
        except KeyboardInterrupt:
            break
        except:
            continue
        else:            
            fluxs.append(f)
            ivars.append(i)            
            spectrum_meta.append(spectrum.__data__)
    fluxs = np.array(fluxs)
    ivars = np.array(ivars)
    
    for each in spectrum_meta:
        for s in spectra:
            if s.spectrum_pk == each["spectrum_pk"]:
                each["sdss4_apogee_member_flags"] = s.sdss4_apogee_member_flags
                break
            
    for i in tqdm(range(len(spectrum_meta))):
        if "Teff" in spectrum_meta[i]:
            continue
        try:
            _, meta = clam_model.fit_stellar_parameters(
                fluxs[i],
                ivars[i],
                basis_weight_interpolator=interpolator,
                random_forest_regressor=rf_model,
                label_names=label_names,
                unscale=unscale,
                full_output=True
            )               
        except KeyboardInterrupt:
            break
        
        except:
            spectrum_meta[i].update(dict(zip(label_names, [np.nan] * len(label_names))))
            spectrum_meta[i]["rchi2"] = np.nan                                    
        else:
            spectrum_meta[i].update(meta["stellar_parameters"])
            spectrum_meta[i]["rchi2"] = meta["rchi2"]
    

    results = Table(rows=spectrum_meta)
    del results["input_spectrum_pks"]
    results.write("20240517_clusters.fits", overwrite=True)
    