import numpy as np
from tqdm import tqdm
from typing import Union, Tuple, Optional
from functools import cached_property
from scipy import optimize as op
from sklearn.decomposition._nmf import _initialize_nmf

import h5py as h5

fp = h5.File("/Users/andycasey/Downloads/dr3-rvs-all.hdf5", "r")

np.random.seed(1234)

N = 10_000



indices = np.sort(np.random.choice(len(fp["source_id"]), N, replace=False))

try:
    flux
except NameError:
    flux = fp["flux"][indices]
    ivar = fp["flux_error"][indices]**-2
    source_id = fp["source_id"][indices]
else:
    print(f"Using pre-loaded data")

EPSILON = np.finfo(np.float32).eps

class CalibratedContinuum(object):

    def __init__(
        self,
        wavelength: np.array,
        deg: int,
        L: Union[float, int],
        regions: Optional[Tuple[Tuple[float, float]]] = None,
        tol: float = 1e-4,
        n_components: int = 8,
        outer_iter: int = 100,
        inner_W_iter: int = 1,
        inner_WH_iter: int = 1,
        init: Optional[str] = None,  
        initial_continuum_scalar = 1,   
        **kwargs
    ):    
        self.L = L
        self.deg = deg
        self.tol = tol
        self.regions = regions or ((0, wavelength.size), )
        self.wavelength = wavelength
        self.n_components = n_components
        self.outer_iter = outer_iter
        self.inner_W_iter = inner_W_iter
        self.inner_WH_iter = inner_WH_iter
        self.init = init
        self.initial_continuum_scalar = initial_continuum_scalar
        return None
    
    @cached_property
    def region_masks(self):
        return _region_masks(self.wavelength, self.regions)
    
    @cached_property
    def continuum_design_matrix(self):
        return _continuum_design_matrix(self.wavelength, self.deg, self.L, self.regions, self.region_masks)
    
    @cached_property
    def n_parameters_per_region(self):
        return max(2 * self.deg + 1, 0)
    
    def _fit_continuum(self, flux, ivar):
        if self.deg is None or self.deg < 0:
            return (None, np.ones_like(flux))

        theta = np.zeros((N, len(self.regions), self.n_parameters_per_region))
            
        N, P = flux.shape
        continuum = np.nan * np.ones_like(flux)
        for i in range(N):            
            for j, mask in enumerate(self.region_masks):
                sj, ej = (j * self.n_parameters_per_region, (j + 1) * self.n_parameters_per_region)
                A = self.continuum_design_matrix[mask, sj:ej]
                MTM = A.T @ (ivar[i, mask][:, None] * A)
                MTy = A.T @ (ivar[i, mask] * flux[i, mask])
                try:
                    theta[i, j] = np.linalg.solve(MTM, MTy)
                except np.linalg.LinAlgError:
                    if np.any(ivar[i, mask] > 0):
                        print(f"Continuum warning on {i}, {j}")
                        continue
                continuum[i, mask] = A @ theta[i, j]        
        return (theta, continuum)
    
    
    def _fit_W_and_theta_by_optimization(self, flux, ivar, H, W_init, theta_init):
        N, P = flux.shape
        theta = np.copy(theta_init)
        W = np.copy(W_init)
        continuum = np.zeros_like(flux)
        
        use = ~self.get_mask(ivar)
        sigma = ivar**-0.5
        chi2, dof = (0, 0)
        
        for i in tqdm(range(N)):
            
            mask = use[i]
            
            def f(_, *p):
                return (
                    (1 - p[:self.n_components] @ H) 
                *   (self.continuum_design_matrix @ p[self.n_components:])
                )[mask]

            p0 = np.hstack([W_init[i], theta_init[i].ravel()])
            
            try:
                p_opt, cov = op.curve_fit(
                    f,
                    None,
                    flux[i][mask],
                    p0=p0,
                    sigma=sigma[i][mask],
                    bounds=self.get_bounds(1)
                )
            except KeyboardInterrupt:
                raise
            except:
                print(f"failed on {i}")
                p_opt = p0
            else:
                W[i] = p_opt[:self.n_components]
                theta[i] = p_opt[self.n_components:].reshape((len(self.regions), self.n_parameters_per_region))
            
            finally:
                continuum[i] = self.continuum_design_matrix @ p_opt[self.n_components:]
                chi2 += np.sum((f(None, *p_opt) - flux[i][mask])**2 * ivar[i][mask])
                dof += np.sum(mask)
                
        return (W, theta, continuum, chi2, dof)


    def _fit_W_and_theta_by_step(self, flux, ivar, X, V, W, H):
    
        N, P = X.shape        
        W = np.copy(W)
        theta = np.zeros((N, len(self.regions), self.n_parameters_per_region))
        continuum = np.ones_like(flux)
        
        chi2, dof = (0, 0)        
        for i in range(N):
            
            # fit W
            for _ in range(self.inner_W_iter):
                # TODO: Use regularization, and possibly switch to use _multiplicative_update function
                W[i] *= ((X[[i]] * V[[i]] @ H.T) / ((V[[i]] * (W[[i]] @ H)) @ H.T))[0]
            
            # fit theta
            model_flux = 1 - (W[[i]] @ H)
            
            if self.deg >= 0:
                continuum_flux = flux[i] / model_flux
                continuum_ivar = ivar[i] * model_flux ** 2
                for j, mask in enumerate(self.region_masks):                
                    sj, ej = (j * self.n_parameters_per_region, (j + 1) * self.n_parameters_per_region)
                    A = self.continuum_design_matrix[mask, sj:ej]
                    MTM = A.T @ (continuum_ivar[0, mask][:, None] * A)
                    MTy = A.T @ (continuum_ivar[0, mask] * continuum_flux[0, mask])
                    try:
                        theta[i, j] = np.linalg.solve(MTM, MTy)
                    except np.linalg.LinAlgError:
                        if np.any(continuum_ivar[0, mask] > 0):
                            print(f"Continuum warning on {i}, {j}")
                            continue
                    continuum[i, mask] = A @ theta[i, j]
        
                    chi2 += np.sum(((flux[i] - continuum[i] * model_flux[0])**2 * ivar[i])[mask])
                    dof += np.sum(mask)
            else:
                chi2 += np.sum(((flux[i] - continuum[i] * model_flux[0])**2 * ivar[i]))
                dof += flux[i].size
                
        assert dof > 0
        
        return (W, theta, continuum, chi2, dof)


    def get_mask(self, ivar):
        N, P = np.atleast_2d(ivar).shape        
        use = np.zeros((N, P), dtype=bool)
        use[:, np.hstack(self.region_masks)] = True
        use *= (ivar > 0)
        return ~use            

    def get_bounds(self, N, component_bounds=(0, +np.inf)):
        C = self.n_components
        A = N * len(self.regions) * (self.n_parameters_per_region)
        return np.vstack([
            np.tile(component_bounds, C).reshape((C, 2)),
            np.tile([-np.inf, +np.inf], A).reshape((-1, 2))
        ]).T            


    def fit(self, flux, ivar):
        
        flux, ivar = _check_and_reshape_flux_ivar(flux, ivar, self.wavelength)
        
        # (1) fit theta (no eigenvectors)
        # (2) fit eigenvectors given theta
        # (3) fit W given H, theta
        # (4) fit theta given W, H
        # (5) go to step 2
        
        theta, continuum = self._fit_continuum(flux, ivar)
        
        continuum *= self.initial_continuum_scalar
        
        X, V = _get_XV(flux, ivar, continuum)
        W, H = _initialize_nmf(X, self.n_components, init=self.init, eps=1e-6, random_state=None)
        
        last_chi2_dof = None
        try:
            with tqdm(total=self.outer_iter) as pb:
                for outer in range(self.outer_iter):    
                    for _ in range(self.inner_WH_iter): # sub-iters
                        _multiplicative_update(X, V, W, H)
                    
                    # For each spectrum, fit W + theta given H
                    W, theta, continuum, chi2, dof = self._fit_W_and_theta_by_step(flux, ivar, X, V, W, H)
                    
                    X, V = _get_XV(flux, ivar, continuum)
                    
                    pb.set_description(f"chi2/dof={chi2/dof:.2e}")
                    pb.update()
                    if last_chi2_dof is None:
                        last_chi2_dof = chi2/dof
                    else:
                        diff = chi2/dof - last_chi2_dof
                        if diff < 0 and np.abs(diff) < self.tol:
                            break
                    
        except KeyboardInterrupt:
            print("Detected KeyboardInterrupt: finishing the fit early")
            None

        return (W, H, theta, continuum, chi2, dof)
            
            
def _multiplicative_update(X, V, W, H, update_H=True, update_W=True, l1_reg_H=0, l2_reg_H=0, l1_reg_W=0, l2_reg_W=0):
    WH = W @ H
    if update_H:
        numerator = ((V.T * X.T) @ W).T
        denominator = ((V.T * WH.T) @ W).T
        if l1_reg_H > 0:
            denominator += l1_reg_H
        if l2_reg_H > 0:
            denominator += l2_reg_H * H
        denominator[denominator == 0] = EPSILON
        H *= numerator / denominator
    
    if update_W:
        numerator = (X * V) @ H.T
        denominator = (V * WH) @ H.T
        if l1_reg_W > 0:
            denominator += l1_reg_W
        if l2_reg_W > 0:
            denominator += l2_reg_W * W
        W *= numerator/denominator
    return None


def _get_XV(flux, ivar, continuum):
    X = 1 - flux / continuum
    V = ivar * continuum**2
    
    is_bad_pixel = (
        (V == 0)
    |   (~np.isfinite(V))
    |   (X < 0)
    |   (~np.isfinite(X))
    )
    X[is_bad_pixel] = V[is_bad_pixel] = 0
    return (X, V)


def _design_matrix(wavelength: np.array, deg: int, L: float) -> np.array:
    scale = 2 * (np.pi / L)
    return np.vstack(
        [
            np.ones_like(wavelength).reshape((1, -1)),
            np.array(
                [
                    [np.cos(o * scale * wavelength), np.sin(o * scale * wavelength)]
                    for o in range(1, deg + 1)
                ]
            ).reshape((2 * deg, wavelength.size)),
        ]
    )


def _continuum_design_matrix(wavelength, deg, L, regions, region_masks):
    n_parameters_per_region = 2 * deg + 1    
    A = np.zeros((wavelength.size, len(regions) * n_parameters_per_region), dtype=float)
    for i, mask in enumerate(region_masks):
        si = i * n_parameters_per_region
        ei = (i + 1) * n_parameters_per_region
        A[mask, si:ei] = _design_matrix(wavelength[mask], deg, L).T        
    return A


def _region_masks(wavelength, regions):
    slices = []
    for region in regions:
        slices.append(np.arange(*wavelength.searchsorted(region), dtype=int))
    return slices


def _check_and_reshape_flux_ivar(flux, ivar, wavelength=None):
    flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
    N1, P1 = flux.shape
    N2, P2 = ivar.shape

    assert (N1 == N2) and (P1 == P2), "`flux` and `ivar` do not have the same shape"
    if wavelength is not None:
        P = len(wavelength)
        assert (P == P1), f"Number of pixels in flux does not match wavelength array ({P} != {P1})"

    bad_pixel = (
        (~np.isfinite(flux))
    |   (~np.isfinite(ivar))
    |   (flux <= 0)
    )
    flux[bad_pixel] = 0
    ivar[bad_pixel] = 0
    return (flux, ivar)


wavelength = np.arange(flux.shape[1])

model = CalibratedContinuum(
    wavelength=wavelength,
    deg=-1,
    L=None,
    n_components=4
)

(W, H, theta, continuum, chi2, dof) = model.fit(flux, ivar)

fig, ax = plt.subplots()
for i in range(model.n_components):
    ax.plot(1 - H[i] / H[i].max() + i)

#source_id
from astropy.table import Table

names, data = (["source_id"], [source_id])
for i in range(W.shape[1]):
    data.append(W[:, i])
    names.append(f"W{i}")
    
t = Table(data=data, names=names)
