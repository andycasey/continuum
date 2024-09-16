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

import matplotlib.pyplot as plt


with open("../applications/harps-sandbox/20240816_train_harps_model.pkl", "rb") as fp:
    λ, label_names, parameters, W, H = pickle.load(fp)

from specutils.utils.wcs_utils import air_to_vac
from astropy import units as u

with open("../applications/harps-sandbox/20240816_train_telfit_model.pkl", "rb") as fp:
    tel_λ, tel_label_names, tel_parameters, tel_W, tel_H = pickle.load(fp)


model = Clam(λ, H, 10 * tel_λ, tel_H)

def fix_vrel_2019_12_04(s):
    s.v_rel = 107.81403 + s.meta["berv"]

options = { 
    "HARPS.2019-12-04T06:28:13.591_e2ds_A.fits": dict(
        read_callback=fix_vrel_2019_12_04,
        fit_kwds=dict(
            vsini=100
        )
    )
}

# O9.5V: ./applications/harps/examples/O9.5V-ms-mu_Col/data/reduced/2019-12-03/HARPS.2019-12-04T06:28:13.591_e2ds_A.fits --> bad RV (-999999)

#for path in ['../applications/harps/alfCenA/calib/2015-10-15/HARPS.2015-10-15T23:18:37.783_e2ds_A.fits']:
#for path in ['../applications/harps/examples/O9.5V-ms-mu_Col/data/reduced/2019-12-03/HARPS.2019-12-04T06:28:13.591_e2ds_A.fits']:
for path in ['../applications/harps/examples/G5-ms-HD_222595/ADP.2014-10-02T10:01:52.103/HARPS.2005-12-20T00:40:22.994_e2ds_A.fits']:
    print(f"Running {path}")

    spectra = SpectrumCollection.read(path, format="HARPS-e2ds", si=30, ei=None)

    spectrum_options = options.get(os.path.basename(path), {})
    if "read_callback" in spectrum_options:
        spectrum_options["read_callback"](spectra)
    
    fit_kwds = spectrum_options.get("fit_kwds", {})

    (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred) = full_result = model.fit(
        spectra, 
        continuum_basis=Sinusoids(9),
        λ_initial=4681,
        **fit_kwds
    )

    fig, (ax, ax_norm) = plt.subplots(2, 1, figsize=(10, 4))
    for i in range(spectra.flux.shape[0]):
        spectrum = spectra.get_order(i)
        ax.plot(spectrum.λ, spectrum.flux, c='k', label=r"$\mathrm{Data}$" if i == 0 else None, zorder=-10)
        ax.plot(spectrum.λ, continuum[i], c="tab:blue", label=r"$\mathrm{Continuum~model}$" if i == 0 else None, zorder=3)
        ax.plot(spectrum.λ, continuum[i] * rectified_model_flux[i], c="tab:orange", label=r"$\mathrm{Stellar~model}$" if i == 0 else None, zorder=-4)
        ax.plot(spectrum.λ, continuum[i] * rectified_telluric_flux[i], c="tab:green", label=r"$\mathrm{Telluric~model}$" if i == 0 else None, zorder=-5)
        ax_norm.plot(spectrum.λ, spectrum.flux / continuum[i], c='k', zorder=-10)
        ax_norm.plot(spectrum.λ, rectified_model_flux[i], c="tab:orange", zorder=-4)
        ax_norm.plot(spectrum.λ, rectified_telluric_flux[i], c="tab:green", lw=1, zorder=-5)
    
    ax.set_title(path)

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