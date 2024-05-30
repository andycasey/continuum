
import numpy as np
import jax
#import jax.numpy as np

from scipy.linalg import lu_factor, lu_solve
from typing import Sequence, Optional
from functools import cached_property
import warnings
from tqdm import tqdm


LARGE = 1e10
SMALL = 1/LARGE

class Clam:

    def __init__(
        self,
        λ,
        H,
        W,
        label_names,
        grid_points,
        deg=3,
        regions=(
            (15161.84316643 - 35, 15757.66995776 + 60),
            (15877.64179911 - 25, 16380.98452330 + 60),
            (16494.30420468 - 30, 16898.18264895 + 60)
        ),
        **kwargs
    ):
        self.λ = λ
        self.H = H
        if W.ndim == 2:
            W = W.reshape((*tuple(map(len, grid_points)), -1))
        self.W = W        
        self.grid_points = grid_points
        self.label_names = label_names
        self.grid_min = np.array(list(map(np.min, grid_points)))
        self.grid_ptp = np.array(list(map(np.ptp, grid_points)))
        
        self.interpolator = RegularGridInterpolator(
            tuple([(g - self.grid_min[i]) / self.grid_ptp[i] for i, g in enumerate(grid_points)]),
            self.W,
            bounds_error=False,
            fill_value=0
        )

        self.regions = regions or [tuple(λ[[0, -1]])]
                        
        # Construct design matrix.
        n_parameters_per_region = 2 * deg + 1
        self.continuum_design_matrix = np.zeros(
            (self.λ.size, len(self.regions) * n_parameters_per_region), 
            dtype=float
        )
        for i, region_slice in enumerate(region_slices(self.λ, self.regions)):
            si = i * n_parameters_per_region
            ei = (i + 1) * n_parameters_per_region
            self.continuum_design_matrix[region_slice, si:ei] = design_matrix(λ[region_slice], deg)
            
        L = len(self.label_names)
        def f(θ):
            W = self.interpolator(θ[:L])
            return (1 - W @ self.H) * (self.continuum_design_matrix @ θ[L:])
                
        self.f = f
        self.g = jax.jacfwd(f)
        return None

    def scale_stellar_parameters(self, p):
        return (p - self.grid_min) / self.grid_ptp
    
    def unscale_stellar_parameters(self, p):
        return p * self.grid_ptp + self.grid_min
    
    @cached_property
    def bounds(self):
        bounds = [(0, 1)] * len(self.label_names)
        bounds += [(-np.inf, +np.inf)] * self.continuum_design_matrix.shape[1]
        return np.array(bounds).T

    
    def get_initial_guess(self, flux, ivar, max_iter=10, large=1e10):

        A = self.continuum_design_matrix
        
        ATCinv = A.T * ivar
        lu, piv = lu_factor(ATCinv @ A)

        full_shape = self.W.shape[:-1]         # The -1 is for the number of components.        
        
        x0 = np.empty((*full_shape, A.shape[1]))
        chi2 = large * np.ones(full_shape)

        # Even though this does not get us to the final edge point in some parameters,
        # NumPy slicing creates a *view* instead of a copy, so it is more efficient.        
        current_step = (np.array(full_shape) - 1) // 2
        current_slice = tuple(slice(0, 1 + end, step) for end, step in zip(full_shape, current_step))
                
        for iter in range(max_iter):                        
            W_slice = self.W[current_slice]
            
            rectified = (1 - W_slice @ H).reshape((-1, self.λ.size))
            
            rta = [np.arange(s)[ss] for s, ss in zip(full_shape, current_slice)]
            
            shape = W_slice.shape[:-1]
            for i, r in enumerate(rectified):
                uri = np.unravel_index(i, shape)
                uai = tuple([rta[j][_] for j, _ in enumerate(uri)])
                if chi2[uai] < large:
                    continue
                x0[uai] = lu_solve((lu, piv), ATCinv @ (flux / r))
                chi2[uai] = np.sum((r * (A @ x0[uai]) - flux)**2 * ivar)
                
            
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
        
        stellar_parameters = tuple(p[i] for p, i in zip(self.grid_points, absolute_index))

        return (stellar_parameters, x0[absolute_index], chi2[absolute_index])
    


    
    def fit(self, flux: Sequence[float], ivar: Sequence[float], p0: Optional[Sequence[float]] = None, use_jac=True, **kwargs):
            
        if p0 is None:
            stellar_parameters, continuum_parameters, chi2 = self.get_initial_guess(flux, ivar)
            p0 = np.hstack([
                self.scale_stellar_parameters(stellar_parameters), 
                continuum_parameters
            ])
        
        if use_jac:

            θ, Σ = op.curve_fit(
                lambda _, *p: self.f(np.array(p)),
                self.λ,
                flux,
                p0=p0,
                jac=lambda _, *p: self.g(np.array(p)),
                sigma=ivar_to_sigma(ivar),
                bounds=self.bounds,
            )
        
        else:
            θ, Σ = op.curve_fit(
                lambda _, *p: self.f(np.array(p)),
                self.λ,
                flux,
                p0=p0,
                sigma=ivar_to_sigma(ivar),
                bounds=self.bounds,
            )            
        y = self.f(θ)
        n_pixels = np.sum(ivar > 0)
        rchi2 = np.sum((flux - y)**2 * ivar) / (n_pixels - len(θ) - 1)
        
        # scale the θ and Σ for stellar params
        L = len(self.label_names)
        θ[:L] = self.unscale_stellar_parameters(θ[:L])
        gp = np.atleast_2d(np.hstack([self.grid_ptp, np.ones(θ.size - L)]))
        Σ *= gp.T @ gp
        
        return (θ, Σ, y, rchi2)
    

def ivar_to_sigma(ivar):
    with np.errstate(divide='ignore'):
        sigma = ivar**-0.5
        sigma[~np.isfinite(sigma)] = LARGE
        return sigma
    
    

def get_next_slice(index, step, shape):    
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
    for index, item in enumerate(all_meta):    
        if item["sdss4_apogee_member_flags"] == 2**18: # M67
            m67.append((index, item))

    results = []
    for index, item in tqdm(m67[:10]):        
        θ, Σ, y, rchi2 = model.fit(flux[index], ivar[index], use_jac=True)
        
        result = dict(zip(model.label_names, θ))
        result["rchi2"] = rchi2
        
        results.append((result, item))
        
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(model.λ, flux[index], c='k')
        ax.plot(model.λ, y, c="tab:red")
        
        print(dict(zip(model.label_names, θ)))
        print(rchi2)        
        raise a
        """
    raise a
    
    from astropy.table import Table
    r = Table(rows=[r[0] for r in results])
    
    
    '''

    '''
    
    raise a
    