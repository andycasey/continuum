import h5py as h5
import os
import numpy as np
from sklearn.decomposition import NMF
import pickle
from tqdm import tqdm

from lsf import rotational_broadening_sparse_matrix


SLICE_ON_C_AND_N = True
SLICE_ON_N_ONLY = False

#NMF_PATH = "20240619_vsini.h5"
NMF_PATH = "20240624_vsini.h5"

λ = 10**(4.179 + 6e-6 * np.arange(8575))


if os.path.exists(NMF_PATH):
    with h5.File(NMF_PATH, "r") as fp:
        label_names = fp["label_names"][:].astype(str)
        W = fp["W"][:]
        H = fp["H"][:]
        grid_points = [fp[f"grid_points/{str(ln)}"][:] for ln in label_names]
        grid_model_flux = fp["grid_model_flux"][:]

else:
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
        
        # build some rotational kernels
        vsinis = np.logspace(0, 2, 6)
        kernels = [rotational_broadening_sparse_matrix(λ, vsini, 0.6) for vsini in tqdm(vsinis[1:])]

        n_pixels = grid_model_flux.shape[-1]
        grid_model_flux = np.clip(grid_model_flux, 0, 1)
        grid_model_flux[~np.isfinite(grid_model_flux)] = 1

        _grid_model_flux = np.zeros((len(vsinis), *grid_model_flux.shape), dtype=float)
        _grid_model_flux[0] = grid_model_flux
        # this is so bad
        for i, kernel in enumerate(tqdm(kernels)):
            for j in range(grid_model_flux.shape[0]):
                for k in range(grid_model_flux.shape[1]):
                    for l in range(grid_model_flux.shape[2]):
                        for m in range(grid_model_flux.shape[3]):
                            _grid_model_flux[i+1, j, k, l, m] = kernel @ grid_model_flux[j, k, l, m]

        # Add a dimension of vsini, evenly sampled in log space because that's what the interpolator needs
        label_names = ["log10_vsini"] + label_names
        grid_points = [np.log10(vsinis)] + grid_points
        grid_model_flux = np.clip(_grid_model_flux, 0, 1)

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

        Y = -np.log(grid_model_flux.reshape((-1, n_pixels)))
        W = nmf_model.fit_transform(Y)
        H = nmf_model.components_

        W = W.reshape((*grid_model_flux.shape[:-1], -1))
        with h5.File(NMF_PATH, "w") as fp:
            fp.create_dataset("label_names", data=label_names)
            fp.create_dataset("W", data=W)
            fp.create_dataset("H", data=H)
            for i, label_name in enumerate(label_names):
                fp.create_dataset(f"grid_points/{label_name}", data=grid_points[i])
            fp.create_dataset("grid_model_flux", data=grid_model_flux)




with open("20240517_spectra.pkl", "rb") as fp:
    flux, ivar, all_meta = pickle.load(fp)

regions = (
    (15120.0, 15820.0),
    (15840.0, 16440.0),
    (16450.0, 16960.0),
)
modes = 7

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

continuum_design_matrix = np.zeros((λ.size, len(regions) * modes), dtype=float)
for i, region_slice in enumerate(region_slices(λ, regions)):
    continuum_design_matrix[region_slice, i*modes:(i+1)*modes] = design_matrix(λ[region_slice], modes)



# Complete the design design_matrix
keep = np.ones(16, dtype=bool)
#keep[0] = False
#keep[2] = False
n_components = np.sum(keep)
A = np.hstack([
    -H[keep].T,
    continuum_design_matrix
])

W_flat = W.reshape((-1, 16))[:, keep]
Λ = np.zeros((A.shape[1], A.shape[1]))
Λ[:n_components, :n_components] = np.cov(W_flat.T)


from tqdm import tqdm
from scipy.spatial.distance import cdist

'''
# fit directly
HTH = (-H @ -H.T)
W_fit = np.zeros_like(W_flat)
for i, f in enumerate(tqdm(grid_model_flux.reshape((-1, 8575)))):
    W_fit[i] = np.linalg.solve(HTH, H @ np.log(f))
'''



rectified_flux = np.nan * np.ones_like(flux)
continuum = np.nan * np.ones_like(rectified_flux)
rchi2 = np.nan * np.ones(flux.shape[0])
X = np.nan * np.ones((flux.shape[0], A.shape[1]))
params = np.nan * np.ones((flux.shape[0], 5))
A_T = A.T
# NOTE TO FUTURE ANDY:
# THIS IS THE CORRECT (or most approximate) TRANSFORMATION. IF YOU FORGET THIS, YOU DIE.
ivar_flux_sq = ivar * flux**2
from scipy import optimize as op
def f(_, *p):
    return A @ p

def jac(_, *p):
    return A


bounds = np.ones((37, 2))
bounds[:n_components, 0] = 0 #np.min(W_flat, axis=0)
bounds[:n_components, 1] = +np.inf #np.max(W_flat, axis=0)
bounds[n_components:, 0] = -np.inf
bounds[n_components:, 1] = +np.inf



bounds = bounds.T

log_flux = np.log(flux)

Xu = np.nan * np.ones((flux.shape[0], A.shape[1]))
Xc = np.nan * np.ones((flux.shape[0], 16))

# Let's check how many end up < 0
for i in tqdm(range(flux.shape[0])):

    Y = log_flux[i]
    finite = np.isfinite(Y) & (ivar[i] > 0)
    Y[~finite] = 0

    Cinv = ivar_flux_sq[i]
    ATCinv = A_T * Cinv
    ATCinvA = ATCinv[:, finite] @ A[finite]
    ATCinvY = ATCinv[:, finite] @ Y[finite]

    Xu[i] = np.linalg.solve(ATCinvA + 1e3 * Λ, ATCinvY)
    continuum = np.exp(A[:, 16:] @ Xu[i][16:])

    Cinv = ivar[i] * (flux[i]/continuum)**2
    Y = np.log(flux[i] / continuum)
    Xc[i] = np.linalg.solve((A_T * Cinv)[:16, finite] @ A[finite, :16], (A_T * Cinv)[:16, finite] @ Y[finite])


    
    '''
    
    # Clip the fits, compute conditioned on the clipped fit
    #for i in tqdm(range(flux.shape[0])):
    rectified_flux = np.exp(A[:, :n_components] @ np.clip(Xu[i][:n_components], 0, np.inf))
    Cinv = np.diag((ivar[i] * rectified_flux**2)[finite])

    ATCinvA = A[finite, n_components:].T @ Cinv @ A[finite, n_components:]
    x = np.linalg.solve(ATCinvA, A[finite, n_components:].T @ Cinv @ (flux[i]/rectified_flux)[finite])

    fig, ax = plt.subplots()
    ax.plot(λ, flux[i], c='k')
    ax.plot(λ, np.exp(A @ Xu[i]) , c='tab:red')
    ax.plot(λ, rectified_flux * (A[:, n_components:] @ x), c='tab:blue')

    if i > 30:
        raise a
    '''

fig, ax = plt.subplots()
ax.scatter(
    W_flat[:, -1],
    W_flat[:, -2],
    s=1
)
ax.scatter(Xc[:, -1], Xc[:, -2], s=1)
ax.scatter(Xu[:, :16][:, -1], Xu[:, :16][:,-2], s=1, c="tab:red")





fig, ax = plt.subplots()
for i in range(16):
    ax.plot(H[i] + i, label=i)


    '''

    p0 = Xu[i].copy()
    p0[:16] = np.clip(Xu[i][:16], 0, np.inf)

    #f = lambda _, *p: np.exp(A[:, :16] @ p[:16]) * (A[:, 16:] @ p[16:])
    f = lambda _, *p: np.exp(A @ p)
    
    x, cov = op.curve_fit(
        f,
        λ,
        flux[i],
        p0=p0,
        sigma=ivar[i]**-0.5,
        bounds=bounds,
    )
    '''




    '''
    if i == 2893:
        fig, (ax, ax_log) = plt.subplots(2, 1)
        ax_log.plot(λ, Y, c='k')
        y_pred = (A @ Xu[i])
        y_pred[~finite] = np.nan
        ax_log.plot(λ, y_pred, c='tab:red')
        ax.plot(λ, flux[i], c='k')
        ax.plot(λ, f(λ, *x), c='tab:blue')
        #ax.plot(λ, f(λ, *p0), c="tab:red")
        #for _ in (ax, ax_log):
        #    _.set_ylim(0, 100)

        raise a
    '''

    fig, ax = plt.subplots()
    ax.plot(λ, flux[i], c='k')
    ax.plot(λ, np.exp(A @ Xu[i]), c='tab:red')
    ax.plot(λ, np.exp(A @ r.x), c="tab:blue")

    if i > 10:
        raise a

    '''
    lsq_linear is giving me some fucked up solutions


    r = op.lsq_linear(
        ATCinvA,
        ATCinvY,
        bounds=bounds,
        method="bvls",
        max_iter=1_000,
    )
    x = r.x

    grid_index = np.unravel_index(np.argmin(cdist(W, np.clip(x[:16], 0, np.inf).reshape(1, -1)).ravel()), grid_model_flux.shape[:-1])
    #for j, (gp, g) in enumerate(zip(grid_points, grid_index)):
    #    params[i, j] = gp[g]

    rectified_flux_ = np.exp(A[:, :16] @ x[:16])
    continuum_ = np.exp(A[:, 16:] @ x[16:])
    chi = (flux[i] - (rectified_flux_ * continuum_))**2 * ivar[i]
    X[i] = x
    rchi2[i] = np.nansum(chi) / (np.sum(finite) - A.shape[1])
    rectified_flux[i] = rectified_flux_
    continuum[i] = continuum_

    fig, ax = plt.subplots()
    ax.plot(λ, flux[i], c='k')
    linalg_rectified_flux  = np.exp(A[:, :16] @ Xu[i][:16])
    linalg_continuum = np.exp(A[:, 16:] @ Xu[i][16:])
    ax.plot(λ, linalg_rectified_flux * linalg_continuum, c='tab:blue')

    ax.plot(λ, rectified_flux[i] * continuum[i], c='tab:red')
    ax.set_title(r.success)

    if i > 9:
        raise a
    '''

fig, ax = plt.subplots()
g = X[:i, :16].flatten()
ax.hist(g, bins=100)

fig, ax = plt.subplots()
ax.scatter(params[:, 3], params[:, 2], c=params[:, 0], alpha=0.1)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])

index = 500

fig, ax = plt.subplots()
ax.plot(λ, flux[index] / continuum[index], c='k')
ax.plot(λ, rectified_flux[index], c='tab:red')

raise a

# Create big array of grid values from grid_points
grid_values = np.zeros((W_flat.shape[0], len(grid_points)))
G = np.meshgrid(*grid_points, indexing="ij")
for i, g in enumerate(G):
    grid_values[:, i] = g.ravel()

fig, axes = plt.subplots(16, len(label_names))
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        ax.scatter(
            grid_values[:, j],
            W_flat[:, i],
            s=1
        )
        ax.set_xlabel(label_names[j])
        ax.set_ylabel(f"W_{i}")


'''
ATA = A.T @ A
W_fit = np.zeros_like(W_flat)
grid_model_flux_flat = grid_model_flux.reshape((-1, 8575))
for i, f in enumerate(tqdm(grid_model_flux_flat)):
    Y = np.log(f)
    W_fit[i] = op.lsq_linear(
        ATA,
        A.T @ Y,
        bounds=bounds,
        method="trf",
        max_iter=1_000
    ).x[:16]
'''

    
mu_grid_values = np.mean(grid_values, axis=0)
ptp_grid_values = np.ptp(grid_values, axis=0)
scaled_grid_values = (grid_values - mu_grid_values) / ptp_grid_values


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=100,
    verbose=10,
    n_jobs=16
)
rf.fit(W_fit, scaled_grid_values)


# Scale grid values?
rfs = []
for i in range(grid_values.shape[1]):
    rfs.append(
        
    )
    rfs[i].fit(W_flat, grid_values[:, i])


params = np.array([rf.predict(X[:, :16]) for rf in rfs]).T

fig, ax = plt.subplots()
ax.scatter(
    params[:, -1],
    params[:, -2],
    c=params[:, 1],
    s=1,
)

rf_self_test = rf.predict(W_flat)

fig, axes = plt.subplots(1, 5)
for i, ax in enumerate(axes):
    ax.scatter(
        grid_values[:, i],
        rf_self_test[:, i],
        s=1
    )
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")





params = rf.predict(X[:, :16]) * ptp_grid_values + mu_grid_values

fig, ax = plt.subplots()
ax.scatter(
    params[:, -1],
    params[:, -2],
    c=params[:, 1],
    s=1
)




lu_piv = lu_factor(ATCinvA + 100 * Λ)




index = 100

Y = np.log(flux[index])
finite = np.isfinite(Y)
Cinv = np.diag(ivar[index][finite] * flux[index][finite]**2)

ATCinv = A[finite].T @ Cinv
ATCinvA = ATCinv @ A[finite]
ATCinvY = ATCinv @ Y[finite]
from time import time


K = 100
t_init = time()
for i in range(K):
    X = np.linalg.solve(ATCinvA + 0 * Λ, ATCinvY)
t_linalg_solve = time() - t_init


t_init = time()
for i in range(K):
    X_lu = lu_solve(lu_piv, ATCinvY)
t_lu = time() - t_init


Y_model = A @ X_lu
rchi2 = np.nansum((Y_model - Y)**2 * ivar[index]) / (np.isfinite(Y).sum() - A.shape[1])
rectified_flux = np.exp(A[:, :16] @ X_lu[:16])
continuum = np.exp(A[:, 16:] @ X_lu[16:])
model = rectified_flux * continuum

def forward_model(p):
    return A @ p

sigma = ivar[index]**-0.5
ln_ivar = (flux[index] * sigma)**-2

finite = np.isfinite(Y * ln_ivar)
def chi2(p):
    return np.sum(((Y - forward_model(p))**2 * ln_ivar)[finite])

result = op.minimize(chi2, X, method="Powell")



raise a


from scipy import optimize as op
bounds = [(0, np.inf)] * H.shape[0]
bounds.extend([(-np.inf, +np.inf)] * (A.shape[1] - H.shape[0]))
t_init = time()
X_lsq = op.lsq_linear(ATCinvA, ATCinvY, bounds=np.array(bounds).T, verbose=1)
t_lsq_linear = K * (time() - t_init)

print(t_linalg_solve)
print(t_lu)
print(t_lsq_linear)

Wi = np.clip(X[:16], 0, np.inf)

# M = mat.dot(vec)
# d = np.einsum('ij,ij->i',mat,mat) + np.inner(vec,vec) -2*M
#return np.sqrt(d)
from scipy.spatial.distance import cdist
gi = np.argmin(cdist(W, Wi.reshape(1, -1)).ravel())
gj = np.argmin(cdist(W, X_lsq.x[:16].reshape(1, -1)).ravel())
print(gi, gj)




fig, axes = plt.subplots(3, 1)
axes[0].plot(λ, flux[index], c='k')

rectified_flux = np.exp(A[:, :16] @ X[:16])
lsq_rectified_flux = np.exp(A[:, :16] @ X_lu[:16])
#grid_rectified_flux = np.exp(A[:, :16] @ W[gi])

continuum = np.exp(A[:, 16:] @ X[16:])
lsq_continuum = np.exp(A[:, 16:] @ X_lu[16:])
axes[0].plot(λ, continuum * rectified_flux, c='tab:red')
#axes[0].plot(λ, lsq_continuum * lsq_rectified_flux, c="tab:blue")
axes[1].plot(λ, flux[index] / continuum, c='k')
axes[1].plot(λ, rectified_flux, c='tab:red')
#axes[1].plot(λ, np.exp(A[:, :16] @ np.clip(X[:16], 0, np.inf)), c="tab:blue")
axes[2].plot(λ, flux[index] / lsq_continuum, c="k")
axes[2].plot(λ, lsq_rectified_flux, c="tab:blue")
axes[2].axhline(1, c="#666666", ls=":", lw=0.5)
axes[1].axhline(1, c="#666666", ls=":", lw=0.5)


'''
ATCinv = A.T * ivar
lu, piv = lu_factor(ATCinv @ A)

x[uai] = lu_solve((lu, piv), ATCinv @ (flux / r))
'''

