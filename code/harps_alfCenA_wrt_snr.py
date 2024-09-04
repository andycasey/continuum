from glob import glob
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from spectrum import Spectrum
from model import Clam, apply_radial_velocity_shift
from continuum_basis import Polynomial, Sinusoids

import matplotlib
#from mpl_utils import mpl_style
#matplotlib.style.use(mpl_style)

def load_harps_e2ds(path, rest_frame=True, mask_na_doublet=True, ignore_last_order=True):

    pattern = path.split("_e2ds_")[0] + "*_bis*.fits"
    bis_path = glob(pattern)[0]
    with fits.open(bis_path) as image:
        bis_rv = image[0].header["ESO DRS BIS RV"]
        berv = image[0].header["ESO DRS BERV"]

    v_shift = -bis_rv + berv
    print(v_shift)

    spectra = []
    with fits.open(path) as image:
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
        K = n_orders - 1 if ignore_last_order else n_orders
        for i in range(K): # ignore last order until we have a good telluric model
            flux = image[0].data[i]
            ivar = 1/image[0].data[i]
            bad_pixel = (flux == 0) | ~np.isfinite(flux) | (flux < 0)
            flux[bad_pixel] = 0
            ivar[bad_pixel] = 0

            if rest_frame:
                λ_order = apply_radial_velocity_shift(λ[i], v_shift)
            else:
                λ_order = λ[i]            

            if mask_na_doublet:
                na_double = (5898 >= λ[i]) * (λ[i] >= 5886)
                ivar[na_double] = 0

            spectra.append(Spectrum(λ_order, flux, ivar, vacuum=False))
            n_pixels += flux.size

    return spectra


with open("../applications/harps-sandbox/20240816_train_harps_model.pkl", "rb") as fp:
    λ, label_names, parameters, W, H = pickle.load(fp)

with open("../applications/harps-sandbox/20240816_train_telfit_model.pkl", "rb") as fp:
    tel_λ, tel_label_names, tel_parameters, tel_W, tel_H = pickle.load(fp)

model = Clam(λ, H)

meta = Table.read("../applications/harps/alfCenA/meta.csv")

#path = "../applications/harps/alfCenA/" + meta["path"][-30]#np.argmin(np.abs(meta["snr"] - 50))]
#path = '../applications/harps/alfCenA/calib/2011-02-27/HARPS.2011-02-28T06:49:35.176_e2ds_A.fits'

# subsample to  only get 1 per snr bin
n_counts = np.zeros(200)
keep = np.zeros(len(meta), dtype=bool)

for i, row in enumerate(meta):
    index = int(row["snr"]) - 1
    if index > (len(n_counts) - 1):
        keep[i] = False
        continue

    if n_counts[index] > 0:
        keep[i] = False
    else:
        keep[i] = True
        n_counts[index] += 1


meta_subset = meta[keep]

from tqdm import tqdm
results = []
for row in tqdm(meta_subset):

    try:
        path = "../applications/harps/alfCenA/" + row["path"]
        if os.path.exists(f"{path}.pkl"):
            with open(f"{path}.pkl", "rb") as fp:
                (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred) = pickle.load(fp)
        else:
                
            spectra = load_harps_e2ds(path)

            (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred) = model.fit(
                spectra, 
                continuum_basis=Sinusoids(5),
                λ_initial=4861,
                v_rel=0.0 # already at rest frame.
            )
            with open(f"{path}.pkl", "wb") as fp:
                pickle.dump((result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred), fp)
            
    except:
        print(f"Failured on {row['path']}")

    else:
        norm_flux = np.array([s.flux / c for s, c in zip(spectra, continuum)]).flatten()
        percentiles = np.percentile(norm_flux, [90, 95, 97.5])
        std = np.std(norm_flux)

        results.append((row["path"], row["snr"], result.status, result.cost, result.optimality, std,  *percentiles))

from astropy.table import Table
t = Table(rows=results, names=("path", "snr", "status", "cost", "optimality", "std_norm_flux", "p_90", "p_95", "p_97.5"))
t.write("alfCenA_stability.csv", overwrite=True)
raise a


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