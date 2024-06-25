
# Here we are going to do just the turn-off:
# 7000 > teff > 4000 and logg > 3
# with a vsini dimension
# try linear and cubic interpolation to see what works
# with a log transformation of the flux

import h5py as h5
import os
import numpy as np
import pickle
from sklearn.decomposition import NMF
from tqdm import tqdm

from lsf import rotational_broadening_sparse_matrix


NMF_PATH = f"{os.path.basename(__file__)[:-3]}.h5"

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


        if sum([SLICE_ON_N_ONLY, SLICE_ON_C_AND_N]) > 1:
            raise ValueError("Only one of SLICE_ON_N_ONLY or SLICE_ON_C_AND_N can be True")
            
        # Take a slice
        c_m = n_m = 0
        c_m_index = list(grid_points[label_names.index("c_m")]).index(c_m)
        n_m_index = list(grid_points[label_names.index("n_m")]).index(n_m)

        label_names = label_names[2:]
        grid_points = grid_points[2:]
        grid_model_flux = grid_model_flux[n_m_index, c_m_index]
    
        # Restrict to a subset
        logg_index = 2
        teff_index = 3
        assert label_names[teff_index] == "Teff"
        assert label_names[logg_index] == "logg"

        logg_mask = grid_points[logg_index] > 3
        teff_mask = (7000 > grid_points[teff_index]) * (grid_points[teff_index] > 4000)

        grid_model_flux = grid_model_flux[:, :, logg_mask]
        grid_model_flux = grid_model_flux[:, :, :, teff_mask]

        grid_points[logg_index] = grid_points[logg_index][logg_mask]
        grid_points[teff_index] = grid_points[teff_index][teff_mask]


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
            tol=1e-5,
            random_state=0,
            verbose=1,
            alpha_W=0.0,
            alpha_H=0.0,
            l1_ratio=0.0,            
        )
        
        nmf_model = NMF(**kwds)

        W = nmf_model.fit_transform(
            -np.log(grid_model_flux.reshape((-1, n_pixels)))
        )
        H = nmf_model.components_
        W = W.reshape((*grid_model_flux.shape[:-1], -1))

        with h5.File(NMF_PATH, "w") as fp:
            fp.create_dataset("label_names", data=label_names)
            fp.create_dataset("W", data=W)
            fp.create_dataset("H", data=H)
            for i, label_name in enumerate(label_names):
                fp.create_dataset(f"grid_points/{label_name}", data=grid_points[i])
            fp.create_dataset("grid_model_flux", data=grid_model_flux)
