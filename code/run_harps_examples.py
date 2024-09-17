from glob import glob
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from spectrum import Spectrum, SpectrumCollection
from spectrum.loaders import get_blaze
from model import Clam, apply_radial_velocity_shift
from continuum_basis import Polynomial, Sinusoids

import matplotlib.pyplot as plt


#with open("../applications/harps-sandbox/20240816_train_harps_model.pkl", "rb") as fp:
#    λ, label_names, parameters, W, H = pickle.load(fp)

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

def apply_blaze(s, p):
    blaze = get_blaze(p)
    s.flux /= blaze
    s.ivar *= blaze**2

def fix_vrel_2019_12_04(s, p):
    apply_blaze(s, p)
    # https://simbad.cds.unistra.fr/simbad/sim-ref?bibcode=2018A%26A...613A..65H
    # v_rad = 111
    s.v_rel = 111 + s.meta["berv"]

def fix_vrel_2011_12_25(s, p):
    s.vrel = -36.16283 + 2.50184

def fix_vrel_2007_07_21(s, p):
    s.vrel = -(-2*5.22852022752343) + 4.7705

options = { 
    "HARPS.2019-12-04T06:28:13.591_e2ds_A.fits": dict(
        read_callback=fix_vrel_2019_12_04,
        fit_kwds=dict(
            vsini=123, # 2005yCat.3244....0G
            continuum_basis=Polynomial(2)
        )
    ),
    "HARPS.2011-12-25T01:33:25.706_e2ds_A.fits": dict(
        fit_kwds=dict(
            vsini=230, # 2005yCat.3244....0G
        )
    ),
    "HARPS.2019-12-04T06:28:13.591_e2ds_A.fits": dict(
        fit_kwds=dict(
            vsini=115, # 2005yCat.3244....0G
        )
    ),
    "HARPS.2005-12-20T00:40:22.994_e2ds_A.fits": dict(
        fit_kwds=dict(
            vsini=3.0, # 2005yCat.3244....0G
        )
    ),
    "HARPS.2009-04-23T09:43:54.824_e2ds_A.fits": dict(
        fit_kwds=dict(
            vsini=30, # 2005yCat.3244....0G
        )
    ),
    "HARPS.2011-12-25T01:33:25.706_e2ds_A.fits": dict(read_callback=fix_vrel_2011_12_25),
    "HARPS.2007-07-21T05:05:54.912_e2ds_A.fits": dict(read_callback=fix_vrel_2007_07_21)
}


def combine_spectrum(λ, flux, ivar, continuum, λ_out):
    O, P = λ.shape
    combined_flux = np.zeros((O, λ_out.size))
    combined_ivar = np.zeros_like(combined_flux)

    for i in range(O):
        combined_flux[i] = np.interp(λ_out, λ[i], flux[i] / continuum[i], left=0, right=0)
        combined_ivar[i] = np.interp(λ_out, λ[i], ivar[i] * continuum[i]**2, left=0, right=0)
    
    finite = np.isfinite(combined_flux) * np.isfinite(combined_ivar)
    combined_flux[~finite] = 0
    combined_ivar[~finite] = 0
    weight = np.sum(combined_ivar, axis=0)
    flux = np.sum(combined_flux * combined_ivar, axis=0) / weight
    ivar = weight
    return (flux, ivar)


default_fit_kwds = dict(continuum_basis=Sinusoids(9), initial_λ=4681)

# "HARPS.2005-12-20T00:40:22.994_e2ds_A.fits']:
paths = (
    glob("../applications/harps/examples/*/*/*/*/*_e2ds_?.fits")
+   glob("../applications/harps/examples/*/*/*_e2ds_?.fits")
)

combined_spectra = dict()

for path in paths:
    print(f"Running {path}")

    spectra = SpectrumCollection.read(path, format="HARPS-e2ds")
    
    spectrum_options = options.get(os.path.basename(path), {})
    if "read_callback" in spectrum_options:
        spectrum_options["read_callback"](spectra, path)
    
    fit_kwds = default_fit_kwds.copy()
    fit_kwds.update(spectrum_options.get("fit_kwds", {}))

    (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred) = full_result = model.fit(spectra, **fit_kwds)

    fig, (ax, ax_norm, ax_combine) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i in range(spectra.flux.shape[0]):
        spectrum = spectra.get_order(i)
        ax.plot(spectrum.λ_vacuum, spectrum.flux, c='k', label=r"$\mathrm{Data}$" if i == 0 else None, zorder=-10)
        ax.plot(spectrum.λ_vacuum, continuum[i], c="tab:blue", label=r"$\mathrm{Continuum~model}$" if i == 0 else None, zorder=3)
        ax.plot(spectrum.λ_vacuum, continuum[i] * rectified_model_flux[i], c="tab:orange", label=r"$\mathrm{Stellar~model}$" if i == 0 else None, zorder=-4)
        ax.plot(spectrum.λ_vacuum, continuum[i] * rectified_telluric_flux[i], c="tab:green", label=r"$\mathrm{Telluric~model}$" if i == 0 else None, zorder=-5)
        ax_norm.plot(spectrum.λ_vacuum, spectrum.flux / continuum[i], c='k', zorder=-10)
        ax_norm.plot(spectrum.λ_vacuum, rectified_model_flux[i], c="tab:orange", zorder=-4)
        ax_norm.plot(spectrum.λ_vacuum, rectified_telluric_flux[i], c="tab:green", lw=1, zorder=-5)
        
    λ_resampled = model.stellar_vacuum_wavelength
    combined_flux, combined_ivar = combine_spectrum(spectra.λ_vacuum, spectra.flux, spectra.ivar, continuum, λ_resampled)

    ax_combine.plot(λ_resampled, combined_flux, c='k', zorder=-10)
    for i in range(spectra.flux.shape[0]):
        spectrum = spectra.get_order(i)
        ax_combine.plot(spectrum.λ_vacuum, rectified_model_flux[i], c="tab:orange", zorder=-4)
        ax_combine.plot(spectrum.λ_vacuum, rectified_telluric_flux[i], c="tab:green", lw=1, zorder=-5)
            
    ax.set_title(path)
    combined_spectra[os.path.basename(path)] = (path, spectra, λ_resampled, combined_flux, combined_ivar, fit_kwds, full_result)

import pickle
with open("harps_examples.pkl", "wb") as fp:
    
# Plot in this order:


raise a
