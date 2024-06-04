
import numpy as np
import warnings
from functools import partial, cached_property
from scipy.linalg import lu_factor, lu_solve
from typing import Sequence, Optional, NamedTuple, Union

from jax import (jit, jacfwd, numpy as jnp, scipy as jsp)
from interpolate import RegularGridInterpolator as JaxRegularGridInterpolator
from scipy.interpolate import RegularGridInterpolator

LARGE = 1e10

    

class ClamFit(NamedTuple):
    
    """A named tuple to store a fit result from the Clam."""

    label_names: Sequence[str]
    stellar_parameters: Sequence[float]
            
    p_init: Sequence[float]
    p_opt: Sequence[float]
    p_cov: np.array
    
    rchi2: float
    dof: int

    model_flux: Sequence[float]
    continuum: Sequence[float]
        
    broadening_kernel: Optional[float] = None


class BaseClam:
    
    """Base class for a constrained linear absorption model."""

    def __init__(
        self,
        λ: Sequence[float],
        H: Sequence[float],
        W: Sequence[float],
        label_names: Sequence[str],
        grid_points: Sequence[Sequence[float]],
        modes: Optional[int] = 7,
        regions: Optional[Sequence[Sequence[float]]] = (
            (15120.0, 15820.0),
            (15840.0, 16440.0),
            (16450.0, 16960.0),
        ),
        **kwargs
    ):
        """        
        :param λ:
            The wavelength grid. This should be a `P`-length array.
            
        :param H:
            The components of the non-negative matrix factorization of the absorption.
            This should be a `(C, P)` shaped array, where `C` is the number of components.
        
        :param W:
            The weights of the components. This should be a `(*N, C)` shape array, where `N` is the
            shape of the grid points.
        
        :param label_names:
            The names of the stellar parameters.
        
        :param grid_points:
            The grid points of the stellar parameters. This should be a `L`-length list of arrays,
            where `L` is the number of stellar parameters. Each list should be ordered.
        
        :param modes:
            The number of Fourier modes to use to model the continuum (in each region).
            
        :param regions:
            The regions to model the continuum. This should be a list of tuples where each tuple
            is a `(start, end)` wavelength region.
        """
        if modes % 2 == 0:
            raise ValueError("modes must be an odd number")
        
        self._λ = λ
        self._H = H
        if W.ndim == 2:
            W = W.reshape((*tuple(map(len, grid_points)), -1))
        self._W = W        
        self._grid_points = grid_points
        self._label_names = tuple(label_names)
        
        self._grid_min = np.array(list(map(np.min, self.grid_points)))
        self._grid_ptp = np.array(list(map(np.ptp, self.grid_points)))

        args = (
            tuple([(g - self._grid_min[i]) / self._grid_ptp[i] for i, g in enumerate(grid_points)]),
            W,            
        )        
        self._interpolator = RegularGridInterpolator(*args,
            bounds_error=False,
            fill_value=0,
        )
        self._jax_interpolator = JaxRegularGridInterpolator(*args, bounds_error=False, fill_value=0)
        self._regions = regions or [tuple(λ[[0, -1]])]
                        
        # Construct design matrix.        
        self._continuum_design_matrix = np.zeros((λ.size, len(self._regions) * modes), dtype=float)
        for i, region_slice in enumerate(region_slices(λ, self._regions)):
            self._continuum_design_matrix[region_slice, i*modes:(i+1)*modes] = design_matrix(λ[region_slice], modes)
            
        self._n_labels = len(self.label_names)        
        return None
    
    # Using Strategy 2 of https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods
    # But hashing is expensive, so we will just never mutate the object. That's why we have a bunch of read-only attributes.
    
    @property
    def λ(self):
        """The wavelength grid."""
        return self._λ
    
    @property
    def H(self):
        """The components of the non-negative matrix factorization of the absorption."""
        return self._H
    
    @property
    def W(self):
        """The weights of the non-negative matrix factorization components."""
        return self._W
    
    @property
    def grid_points(self):
        """The grid points of the stellar parameters."""
        return self._grid_points
    
    @property
    def label_names(self):
        """The names of the stellar parameters."""
        return self._label_names
    
    @property
    def grid_min(self):
        """The minimum values of the grid points."""
        return self._grid_min
    
    @property
    def grid_ptp(self):
        """The peak-to-peak values of the grid points."""
        return self._grid_ptp
    
    @property
    def interpolator(self):
        """The interpolator for the weights of the non-negative matrix factorization components."""
        return self._interpolator

    @property
    def continuum_design_matrix(self):
        """The design matrix for the continuum."""
        return self._continuum_design_matrix
    
    @property
    def bounds(self):
        """Bounds on the model parameters."""
        return self._bounds
    
    @property
    def regions(self):
        """The regions to model the continuum."""
        return self._regions
    
    @property
    def n_labels(self):
        """The number of stellar parameter labels."""
        return self._n_labels  
    
    @cached_property
    def bounds(self):
        """Bounds on the model parameters."""
        bounds = [(0, 1)] * self._n_labels
        bounds += [(-np.inf, +np.inf)] * self.continuum_design_matrix.shape[1]
        return np.array(bounds).T          
    
    # TODO: Computing the hash is too expensive. We should just never mutate the object.
    # Note: If you ever experience weirdness,.. this is the place to start debugging.
    def __hash__(self):
        return id(self)    
    
    
    def scale_stellar_parameters(self, stellar_parameters: Sequence[float]) -> Sequence[float]:
        """Scale the stellar parameters to the range [0, 1] for the interpolator."""
        return (stellar_parameters - self.grid_min) / self.grid_ptp
    
    
    def unscale_stellar_parameters(self, scaled_stellar_parameters: Sequence[float]) -> Sequence[float]:
        """Unscale the stellar parameters from the range [0, 1] for the interpolator."""
        return scaled_stellar_parameters * self.grid_ptp + self.grid_min
        
        
    def zero_absorption_initial_guess(self, flux: Sequence[float], ivar: Sequence[float]) -> Sequence[float]:
        """
        Return an initial guess of the model parameters assuming zero absorption.
        
        The continuum parameters are set to zero, and the stellar parameters are set 
        to the mid-point in each dimension.
        
        :param flux:
            The observed flux.
        
        :param ivar:
            The inverse variance of the observed flux.     
        
        :returns:
            The initial guess of the model parameters.  
        """
        A = self.continuum_design_matrix
        ATCinv = A.T * ivar
        return np.hstack([
            0.5 * np.zeros(self.n_labels),
            np.linalg.solve(ATCinv @ A, ATCinv @ flux)
        ])
    
    
    def initial_guess(
        self,
        flux: Sequence[float],
        ivar: Sequence[float],
        max_iter: Optional[int] = 10,
        max_step_size: Optional[Union[int, Sequence[int]]] = 8,
        full_output: Optional[bool] = False,
        verbose: Optional[bool] = False,        
    ) -> Sequence[float]:
        """
        Step through the grid of stellar parameters to find a good initial guess.
        
        :param flux:
            The observed flux.
        
        :param ivar:
            The inverse variance of the observed flux.
        
        :param max_iter: [optional]
            The maximum number of iterations to take.
            
        :param max_step_size: [optional]
            The maximum step size to allow, either as a single value per label dimension, or as a
            tuple of integers. If `None`, then no restriction is made on the maximum possible step,
            which means that in the first iteration the only sampled values in one label might be
            the: smallest value, highest value, and the middle value. Whereas if given, this sets
            the maximum possible step size per dimension. This adds computational cost, but can
            be useful to avoid getting stuck in local minima.
                            
        :param full_output: [optional]
            Return a two-length tuple containing the initial guess and a dictionary of metadata.
            
        :param verbose: [optional]
            Print verbose output.
        
        :returns:
            The initial guess of the model parameters. If `full_output` is true, then an
            additional dictionary of metadata is returned.
        """
        
        A = self.continuum_design_matrix
        
        ATCinv = A.T * ivar
        lu, piv = lu_factor(ATCinv @ A)

        full_shape = self.W.shape[:-1] # The -1 is for the number of components.        
        rta_indices = tuple(map(np.arange, full_shape))
        
        x = np.empty((*full_shape, A.shape[1]))
        chi2 = LARGE * np.ones(full_shape)
        n_evaluations = 0
        
        if max_step_size is None:
            max_step_size = full_shape

        # Even though this does not get us to the final edge point in some parameters,
        # NumPy slicing creates a *view* instead of a copy, so it is more efficient.                
        current_step = np.clip(
            (np.array(full_shape) - 1) // 2,
            0,
            max_step_size
        )
        if verbose:
            print(f"initial step: {current_step}")
                        
        current_slice = tuple(slice(0, 1 + end, step) for end, step in zip(full_shape, current_step))
                
        for n_iter in range(1, 1 + max_iter):      
            W_slice = self.W[current_slice]
            
            rectified = (1 - W_slice @ H).reshape((-1, self.λ.size))
            
            # map relative indices to absolute ones
            rta = [rtai[ss] for rtai, ss in zip(rta_indices, current_slice)]
            
            shape = W_slice.shape[:-1]
            for i, r in enumerate(rectified):
                uri = np.unravel_index(i, shape)
                uai = tuple(rta[j][_] for j, _ in enumerate(uri))
                if chi2[uai] < LARGE:
                    # We have computed this solution already.
                    continue
                x[uai] = lu_solve((lu, piv), ATCinv @ (flux / r))
                chi2[uai] = np.sum((r * (A @ x[uai]) - flux)**2 * ivar)
                n_evaluations += 1
                            
            # Get next slice
            relative_index = np.unravel_index(np.argmin(chi2[current_slice]), shape)
            absolute_index = tuple([rta[j][_] for j, _ in enumerate(relative_index)])
            
            if verbose:
                print("current_slice: ", current_slice)
                for i, cs in enumerate(current_slice):
                    print(f"\t{self.label_names[i]}: {self.grid_points[i][cs]}")
                print(f"n_iter={n_iter}, chi2={chi2[absolute_index]}, x={[p[i] for p, i in zip(self.grid_points, absolute_index)]}")
                print(f"absolute index {absolute_index} -> {dict(zip(self.label_names, [p[i] for p, i in zip(self.grid_points, absolute_index)]))}")
                
            next_step = np.clip(
                np.clip(current_step // 2, 1, full_shape),
                0,
                max_step_size
            )
                
            next_slice = get_next_slice(absolute_index, next_step, full_shape)
            if verbose:
                print("next_slice", next_slice)

            if next_slice == current_slice and max(next_step) == 1:
                if verbose:
                    print("stopping")
                break
            
            current_step, current_slice = (next_step, next_slice)
        else:
            warnings.warn(f"Maximum iterations reached ({max_iter}) for initial guess")

        p_init = np.hstack([
            self.scale_stellar_parameters([p[i] for p, i in zip(self.grid_points, absolute_index)]),
            x[absolute_index]
        ])
        if not full_output:
            return p_init
        
        meta = dict(
            x=x,
            chi2=chi2,
            n_iter=n_iter,
            n_evaluations=n_evaluations, 
            min_chi2=chi2[absolute_index],
            large=LARGE
        )
        return (p_init, meta)


    def fit(self, flux: Sequence[float], ivar: Sequence[float], p_init: Optional[Sequence[float]] = None, **kwargs):
        """
        Fit the model to the data.
        
        :param flux:
            The observed flux.
        
        :param ivar:
            The inverse variance of the observed flux.
        
        :param p_init: [optional]
            The initial guess for the model parameters.
        
        :returns:
            A dictionary.
        """
            
        if p_init is None:
            p_init = self.initial_guess(flux, ivar)            
        
        kwds = dict(check_finite=False, absolute_sigma=True, bounds=self.bounds)
        kwds.update(**kwargs)
        
        θ, Σ = op.curve_fit(
            lambda _, *p: self(np.array(p)),
            None,
            flux,
            p0=p_init,
            jac=lambda _, *p: self.jacobian(np.array(p)),
            sigma=ivar_to_sigma(ivar),
            **kwds
        )    
        y = self(θ)
        n_pixels = np.sum(ivar > 0)
        dof = n_pixels - len(θ) - 1
        rchi2 = np.sum((flux - y)**2 * ivar) / dof
        
        # scale the θ and Σ for stellar params.
        θ[:self.n_labels] = self.unscale_stellar_parameters(θ[:self.n_labels])        
        scale = np.atleast_2d(np.hstack([self.grid_ptp, np.ones(θ.size - self.n_labels)]))
        scale = scale.T @ scale
        assert scale.size > 1
        Σ *= scale
        
        return dict(
            p_init=p_init,
            p_opt=θ,
            p_cov=Σ,
            rchi2=rchi2,
            dof=dof,
            model_flux=y,
            continuum=self.continuum_design_matrix @ θ[self.n_labels:self.n_labels + self.continuum_design_matrix.shape[1]],
            label_names=self.label_names,
            stellar_parameters=θ[:self.n_labels]
        )
                

    def predict(self, θ):
        """
        Predict the flux given the stellar parameters and the continuum parameters.
        
        This is a wrapper around the `__call__` method which checks whether the 
        first `L` labels are scaled or not. If any are outside the bounds of (0, 1)
        then they will be scaled appropriately.
        
        :param θ:
            The model parameters.
        """
        not_scaled = (
            np.any(θ[:self.n_labels] > 1)
        or  np.any(θ[:self.n_labels] < 0)
        )
        if not_scaled:
            θ = np.copy(θ)
            θ[:self.n_labels] = self.scale_stellar_parameters(θ[:self.n_labels])
        return self(θ)





class Clam(BaseClam):
    
    """A constrained linear absorption model."""
    
    def __call__(self, θ: Sequence[float]):
        """
        Predict the flux given the stellar parameters and the continuum parameters.
        
        :param θ: 
            The model parameters. This should be a `N`-length sequence where the first `L` elements
            are the scaled stellar parameters (i.e., between 0 and 1), and the next `C` elements
            are the continuum parameters. If rotational broadening is being fit, then this is the
            last element of sequence. 
            
            You can check the values of `L` (number of stellar parameter labels) using the 
            `n_labels` attribute, and you can check the number of continuum parameters using the 
            `continuum_design_matrix` attribute, as that matrix should have shape `(P, C)` where 
            `P` is the number of pixels.
        """
        return (1 - self.interpolator(θ[:self.n_labels], method="slinear") @ self.H) * (self.continuum_design_matrix @ θ[self.n_labels:])
    
        
    
    def jacobian(self, θ: Sequence[float]):
        """
        Compute the Jacobian of the predicted model with respect to the model parameters.
        
        :param θ: 
            The model parameters. This should be a `N`-length sequence where the first `L` elements
            are the scaled stellar parameters (i.e., between 0 and 1), and the next `C` elements
            are the continuum parameters. If rotational broadening is being fit, then this is the
            last element of sequence. 
            
            You can check the values of `L` (number of stellar parameter labels) using the 
            `n_labels` attribute, and you can check the number of continuum parameters using the 
            `continuum_design_matrix` attribute, as that matrix should have shape `(P, C)` where 
            `P` is the number of pixels.
        """
        
        dW_dtheta = []
        for axis in range(self.n_labels):
            nu = np.zeros(self.n_labels, dtype=int)
            nu[axis] = 1
            dW_dtheta.append(self.interpolator(θ[:self.n_labels], nu=nu, method="slinear")) 
        
        dW_dtheta = np.array(dW_dtheta).reshape((self.n_labels, -1))
        
        # should be shape (8575, -1)
        rectified_flux = (1 - self.interpolator(θ[:self.n_labels], method="slinear") @ self.H).reshape((-1, 1))
        continuum = self.continuum_design_matrix @ θ[self.n_labels:]
        
        return np.hstack([
            (-dW_dtheta @ self.H).T * continuum[:, np.newaxis],
            rectified_flux * self.continuum_design_matrix
        ])
        
    def _jax_call(self, θ: Sequence[float]):
        return (1 - self._jax_interpolator(θ[:self.n_labels]) @ self.H) * (self.continuum_design_matrix @ θ[self.n_labels:])


    @partial(jit, static_argnums=(0,))
    def _jax_jacobian(self, θ: Sequence[float]):
        """
        Compute the Jacobian of the predicted model with respect to the model parameters.
        
        :param θ: 
            The model parameters. This should be a `N`-length sequence where the first `L` elements
            are the scaled stellar parameters (i.e., between 0 and 1), and the next `C` elements
            are the continuum parameters. If rotational broadening is being fit, then this is the
            last element of sequence. 
            
            You can check the values of `L` (number of stellar parameter labels) using the 
            `n_labels` attribute, and you can check the number of continuum parameters using the 
            `continuum_design_matrix` attribute, as that matrix should have shape `(P, C)` where 
            `P` is the number of pixels.
        """
        return jacfwd(self._jax_call)(θ)
            
        
        '''
            def jacobian(theta):
                # compute rectified flux and continuum
                rectified_flux = 1 - theta[:C] @ self.components
                continuum = A_slice @ theta[C:-1]
                
                return np.hstack([
                    (A_slice @ theta[C:-1, None]) * mBVT_inv_sigma, # dy/dW (basis weights)
                    (1 + mBVT @ theta[:C, None]) * A_inv_sigma, # dy/dphi (continuum)
                    op.approx_fprime(theta[-1], _apply_rotation, vsini_epsilon, rectified_flux, continuum)
                ])                
    
        '''
    
    def fit(self, flux: Sequence[float], ivar: Sequence[float], p_init: Optional[Sequence[float]] = None, **kwargs):
        """
        Fit the model to the data.
        
        :param flux:
            The observed flux.
        
        :param ivar:
            The inverse variance of the observed flux.
        
        :param p_init: [optional]
            The initial guess for the model parameters.
        
        :returns:
            A `ClamFit` named tuple.
        """        
        return ClamFit(**super(Clam, self).fit(flux, ivar, p_init=p_init, **kwargs))



def ivar_to_sigma(ivar: Sequence[float]) -> Sequence[float]:
    """
    Convert the inverse variance to standard deviation.
    
    :param ivar:
        The inverse variance.
    
    :returns:
        The standard deviation.
    """
    with np.errstate(divide='ignore'):
        sigma = ivar**-0.5
        sigma[~np.isfinite(sigma)] = LARGE
        return sigma
    
    
def get_next_slice(index: Sequence[float], step: Sequence[float], shape: Sequence[float]):
    """
    Get the next slice given the current position, step, and dimension shapes.
    
    :param index:
        The best current index.
    
    :param step:
        The step size in each dimension.
    
    :param shape:
        The shape of the grid.
    
    :returns:
        A tuple of slices for the next iteration.
    """
    next_slice = []
    for center, step, end in zip(index, step, shape):
        start = center - step
        stop = center + step + 1                
        if start < 0:
            offset = -start
            start += offset
            stop += offset
        elif stop > end:
            offset = stop - (end + 1)
            start -= offset
            stop -= offset
        next_slice.append(slice(start, stop, step))
    return tuple(next_slice)

    
def region_slices(λ, regions):
    slices = []
    for region in np.array(regions):
        si, ei = λ.searchsorted(region)
        slices.append(slice(si, ei + 1))
    return slices    


def design_matrix(λ: np.array, modes: int) -> np.array:
    #L = 1300.0
    #scale = 2 * (np.pi / (2 * np.ptp(λ)))
    scale = np.pi / np.ptp(λ)
    return np.vstack(
        [
            np.ones_like(λ).reshape((1, -1)),
            np.array(
                [
                    [np.cos(o * scale * λ), np.sin(o * scale * λ)]
                    for o in range(1, (modes - 1) // 2 + 1)
                ]
            ).reshape((modes - 1, λ.size)),
        ]
    ).T



if __name__ == "__main__":
    
    import os
    import h5py as h5
    import pickle
    from sklearn.decomposition import NMF

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
        #absorption[~np.isfinite(absorption)] = 0
        
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
    
    model = Clam(
        λ=10**(4.179 + 6e-6 * np.arange(8575)),
        H=H,
        W=W,
        label_names=label_names,
        grid_points=grid_points,
    )
    
    import pickle
    with open("20240517_spectra.pkl", "rb") as fp:
        flux, ivar, all_meta = pickle.load(fp)
    
    m67 = []
    pleades = []
    ngc188 = []
    for index, item in enumerate(all_meta):    
        if item["sdss4_apogee_member_flags"] == 2**18: # M67
            m67.append((index, item))
        if item["sdss4_apogee_member_flags"] == 2**17: # NGC188
            ngc188.append((index, item))
        if item["sdss4_apogee_member_flags"] == 2**20: # Pledas
            pleades.append((index, item))

    
    X = model.initial_guess(flux[0], ivar[0])
    
    f1 = model(X)
    g1 = model.jacobian(X)
    
    f2 = model._jax_call(X)
    g2 = model._jax_jacobian(X)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(g1.flatten(), g2.flatten())
    
    raise a

    pleiades_results = []
    for index, item in tqdm(pleades):        
        
        result = model.fit(flux[index], ivar[index])
        
        pleiades_results.append((index, result, item))


    e_m_h = [e[1].p_cov[3, 3]**0.5 for e in pleiades_results]
    t = Table(rows=[e[1].stellar_parameters for e in pleiades_results], names=list(model.label_names))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    scat = ax.scatter(t["Teff"], t["logg"], c=e_m_h, s=5)# vmin=0, vmax=5)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    
    cbar = plt.colorbar(scat)
    
    raise a

    results = []
    for index, item in tqdm(m67):
        result = model.fit(flux[index], ivar[index])
        results.append((index, result, item))
        
        #fig, ax = plt.subplots()
        #ax.plot(model.λ, flux[index], c='k')
        #ax.plot(model.λ, result.model_flux, c="tab:red")
        

    t = Table(
        np.array([r[1].stellar_parameters for r in results]), 
        names=model.label_names
    )

    rchi2 = np.array([r[1].rchi2 for r in results])
    vrot = np.array([r[1].broadening_kernel for r in results])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    scat = ax.scatter(t["Teff"], t["logg"], c=rchi2, s=5)# vmin=0, vmax=5)#t["m_h"], s=5)# vmin=0, vmax=5)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    
    cbar = plt.colorbar(scat)
    

    pleiades_results = []
    for index, item in tqdm(pleades):        
        
        result = model.fit(flux[index], ivar[index])
        
        pleiades_results.append((index, result, item))


        
    t = Table(rows=[e[1].stellar_parameters for e in pleiades_results], names=list(model.label_names))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    scat = ax.scatter(t["Teff"], t["logg"], c=t["m_h"], s=5)# vmin=0, vmax=5)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    
    cbar = plt.colorbar(scat)
    
    raise a
    
    model = Clam(
        λ=10**(4.179 + 6e-6 * np.arange(8575)),
        H=H,
        W=W,
        label_names=label_names,
        grid_points=grid_points,
    )
        
    
    m67_results = []
    for index, item in tqdm(m67):
        result = model.fit(flux[index], ivar[index])
        
        m67_results.append(np.hstack([result.stellar_parameters, result.broadening_kernel]))
    
    t = Table(rows=m67_results, names=list(model.label_names) + ["vsini"])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    scat = ax.scatter(t["Teff"], t["logg"], c=t["m_h"], s=5)# vmin=0, vmax=5)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    
    cbar = plt.colorbar(scat)
        
    raise a
    
    from astropy.table import Table
    r = Table(rows=[r[0] for r in results])
    
    
    '''

    '''
    
    raise a
    