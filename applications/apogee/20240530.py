
import numpy as np
import jax
from functools import partial, cached_property
from scipy.linalg import lu_factor, lu_solve
from typing import Sequence, Optional
import warnings

# TODO: move furniture around
from interpolate import RegularGridInterpolator
#from jax_ndimage import gaussian_filter1d
import jax.numpy as jnp

LARGE = 1e10

class BaseClam:

    def __init__(
        self,
        λ,
        H,
        W,
        label_names,
        grid_points,
        modes=7,
        regions=(
            (15161.84316643 - 35, 15757.66995776 + 60),
            (15877.64179911 - 25, 16380.98452330 + 60),
            (16494.30420468 - 30, 16898.18264895 + 60)
        ),
        **kwargs
    ):
        """
        A constrained linear absorption model.
        
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
        
        :param modes: [optional]
            The number of Fourier modes to use to model the continuum (in each region).
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
        
        self._interpolator = RegularGridInterpolator(
            tuple([(g - self._grid_min[i]) / self._grid_ptp[i] for i, g in enumerate(grid_points)]),
            W,
            bounds_error=False,
            fill_value=0
        )

        regions = regions or [tuple(λ[[0, -1]])]
                        
        # Construct design matrix.        
        self._continuum_design_matrix = np.zeros((λ.size, len(regions) * modes), dtype=float)
        for i, region_slice in enumerate(region_slices(λ, regions)):
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

    @partial(jax.jit, static_argnums=(0,))
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
        return jax.jacfwd(self.__call__)(θ)
    
    
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
        full_output: Optional[bool] = False
    ) -> Sequence[float]:
        """
        Step through the grid of stellar parameters to find a good initial guess.
        
        :param flux:
            The observed flux.
        
        :param ivar:
            The inverse variance of the observed flux.
        
        :param max_iter: [optional]
            The maximum number of iterations to take.
                
        :param full_output: [optional]
            Return a two-length tuple containing the initial guess and a dictionary of metadata.
        
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

        # Even though this does not get us to the final edge point in some parameters,
        # NumPy slicing creates a *view* instead of a copy, so it is more efficient.        
        current_step = (np.array(full_shape) - 1) // 2
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
            
            next_step = np.clip(current_step // 2, 1, full_shape)            
            next_slice = get_next_slice(absolute_index, next_step, full_shape)

            if next_slice == current_slice and max(next_step) == 1:
                break
            
            current_step, current_slice = (next_step, next_slice)
        else:
            warnings.warn(f"Maximum iterations reached ({max_iter}) for initial guess")

        p0 = np.hstack([
            self.scale_stellar_parameters([p[i] for p, i in zip(self.grid_points, absolute_index)]),
            x[absolute_index]
        ])
        if not full_output:
            return p0
        
        meta = dict(
            x=x,
            chi2=chi2,
            n_iter=n_iter,
            n_evaluations=n_evaluations, 
            min_chi2=chi2[absolute_index],
            large=LARGE
        )
        return (p0, meta)


    def fit(self, flux: Sequence[float], ivar: Sequence[float], p0: Optional[Sequence[float]] = None, **kwargs):
        """
        Fit the model to the data.
        
        :param flux:
            The observed flux.
        
        :param ivar:
            The inverse variance of the observed flux.
        
        :param p0: [optional]
            The initial guess for the model parameters.
        
        :returns:
            A five-length tuple containing:
            - the best-fit stellar parameters,
            - the best-fit model parameters (including unscaled stellar parameters), 
            - the covariance matrix of the best-fit parameters, 
            - the model flux, and 
            - the reduced chi-squared value.
        """
            
        if p0 is None:
            p0 = self.initial_guess(flux, ivar)            
        
        kwds = dict(check_finite=False, absolute_sigma=True, bounds=self.bounds)
        kwds.update(**kwargs)
        
        θ, Σ = op.curve_fit(
            lambda _, *p: self(np.array(p)),
            None,
            flux,
            p0=p0,
            jac=lambda _, *p: self.jacobian(np.array(p)),
            sigma=ivar_to_sigma(ivar),
            **kwds
        )    
        y = self(θ)
        n_pixels = np.sum(ivar > 0)
        rchi2 = np.sum((flux - y)**2 * ivar) / (n_pixels - len(θ) - 1)
        
        # scale the θ and Σ for stellar params?
        stellar_parameters = self.unscale_stellar_parameters(θ[:self.n_labels])        
        return (stellar_parameters, θ, Σ, y, rchi2)


class Clam(BaseClam):
    
    """A constrained linear absorption model without rotational broadening."""
    
    @partial(jax.jit, static_argnums=(0,))
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
        W = self.interpolator(θ[:self.n_labels])
        return (1 - W @ self.H) * (self.continuum_design_matrix @ θ[self.n_labels:])
    
    

class RotationallyBroadenedClam(BaseClam):
    
    """A constrained linear absorption model with rotational broadening."""
    
    _initial_filter_sigma = 10.0
    
    @partial(jax.jit, static_argnums=(0,))
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
        W = self.interpolator(θ[:self.n_labels])
        rectified_flux = jnp.convolve(1 - W @ self.H, jnp.array([θ[-1]]))
        return rectified_flux * (self.continuum_design_matrix @ θ[self.n_labels:-1])
        

    @cached_property
    def bounds(self):
        """Bounds on the model parameters."""
        return np.hstack([
            super(RotationallyBroadenedClam, self).bounds, 
            np.atleast_2d([0, np.inf]).T
        ])
        
        
    def zero_absorption_initial_guess(self, flux: Sequence[float], ivar: Sequence[float]) -> Sequence[float]:
        """
        Return an initial guess of the model parameters assuming zero absorption.
        
        The continuum parameters are set to zero, and the stellar parameters are set
        to the mid-point in each dimension. The last parameter is the broaadening, which is initially set to 10.
        
        :param flux:
            The observed flux.
        
        :param ivar:
            The inverse variance of the observed flux.
        """
        x = super(RotationallyBroadenedClam, self).zero_absorption_initial_guess(flux, ivar)
        return np.hstack([x, self._initial_filter_sigma])
    
    
    def initial_guess(
        self,
        flux: Sequence[float],
        ivar: Sequence[float],
        max_iter: Optional[int] = 10,
        full_output: Optional[bool] = False
    ):
        """
        Step through the grid of stellar parameters to find a good initial guess.
        
        :param flux:
            The observed flux.
        
        :param ivar:
            The inverse variance of the observed flux.
        
        :param max_iter: [optional]
            The maximum number of iterations to take.
                
        :param full_output: [optional]
            Return a two-length tuple containing the initial guess and a dictionary of metadata.
        
        :returns:
            The initial guess of the model parameters. If `full_output` is true, then an
            additional dictionary of metadata is returned.
        """        
        x0, meta = super(RotationallyBroadenedClam, self).initial_guess(flux, ivar, max_iter=max_iter, full_output=True)
        x0 = np.hstack([x0, self._initial_filter_sigma])
        return (x0, meta) if full_output else x0
    

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
    deg = (modes - 1) // 2
    scale = np.pi / np.ptp(λ)
    return np.vstack(
        [
            np.ones_like(λ).reshape((1, -1)),
            np.array(
                [
                    [np.cos(o * scale * λ), np.sin(o * scale * λ)]
                    for o in range(1, deg + 1)
                ]
            ).reshape((2 * deg, λ.size)),
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
        
    from interpolate import RegularGridInterpolator
        
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
    
    model = RotationallyBroadenedClam(
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
    for index, item in enumerate(all_meta):    
        if item["sdss4_apogee_member_flags"] == 2**18: # M67
            m67.append((index, item))

    results = []
    for index, item in tqdm(m67[:100]):        
        stellar_parameters, θ, Σ, y, rchi2 = model.fit(flux[index], ivar[index])
        
        result = dict(zip(model.label_names, stellar_parameters))
        result["vsini"] = θ[-1]
        result["rchi2"] = rchi2
        
        results.append(result)
        #(result, item))
        
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(model.λ, flux[index], c='k')
        ax.plot(model.λ, y, c="tab:red")
        
        print(dict(zip(model.label_names, θ)))
        print(rchi2)        
        raise a
        """
        
    t = Table(rows=results)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    scat = ax.scatter(t["Teff"], t["logg"], c=t["vsini"], s=5)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    
    cbar = plt.colorbar(scat)
    
    raise a
    
    from astropy.table import Table
    r = Table(rows=[r[0] for r in results])
    
    
    '''

    '''
    
    raise a
    