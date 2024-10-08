from glob import glob
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from spectrum import Spectrum, SpectrumCollection
from model import Clam, apply_radial_velocity_shift
from continuum_basis import Polynomial, Sinusoids

import matplotlib
#from mpl_utils import mpl_style
#matplotlib.style.use(mpl_style)


with open("20240916_train_harps_lsf_model.pkl", "rb") as fp:
    λ, label_names, parameters, W, H = pickle.load(fp)


from specutils.utils.wcs_utils import air_to_vac
from astropy import units as u

with open("../applications/harps-sandbox/20240816_train_telfit_model.pkl", "rb") as fp:
    tel_λ, tel_label_names, tel_parameters, tel_W, tel_H = pickle.load(fp)
    tel_λ *= 10
    

with open("20240916_telluric_model.pkl", "rb") as fp:
    tel_λ, tel_W, tel_H, tel_meta = pickle.load(fp)

keep = np.ones(10, dtype=bool)
for index in (0, 2, 3, 5, 7, 9):
    keep[index] = False

model = Clam(λ, H, tel_λ, tel_H[keep])


meta = Table.read("../applications/harps/alfCenA/meta.csv")


# subsample to  only get 1 per snr bin
n_counts = np.zeros(300)
keep = np.zeros(len(meta), dtype=bool)

for i, row in enumerate(meta):
    if row["path"].endswith("_B.fits"):
        continue

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

suffix = "_20240930"
from tqdm import tqdm
results = []
for row in tqdm(meta_subset):

    try:
        path = "../applications/harps/alfCenA/" + row["path"]

        if os.path.exists(f"{path}{suffix}.pkl"):
            print(f"done {path}")
            continue
            #with open(f"{path}{suffix}.pkl", "rb") as fp:
            #    (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred) = pickle.load(fp)
        else:
                
            spectra = SpectrumCollection.read(path)

            (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred) = model.fit(
                spectra, 
                continuum_basis=Sinusoids(9),
                initial_λ=4861,
            )
            with open(f"{path}{suffix}.pkl", "wb") as fp:
                pickle.dump((result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred), fp)
            
    except:
        print(f"Failured on {row['path']}")
        continue

raise a

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
