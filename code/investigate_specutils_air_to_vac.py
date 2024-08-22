from glob import glob
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle

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

with open("cont.pkl", "rb") as fp:
    continuum = pickle.load(fp)

with open("model.pkl", "rb") as fp:
    model_vac_wl, model_flux = pickle.load(fp)

path = '../applications/harps/alfCenA/calib/2011-02-27/HARPS.2011-02-28T06:49:35.176_e2ds_A.fits'

spectra = load_harps_e2ds(path)

spectrum = spectra[35]

fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
axes[0].set_title("No shift to data. Model in vacuum wavelengths. Data in air.")
axes[0].plot(spectrum.λ, spectrum.flux / continuum, c='k', label="Data")
axes[0].plot(model_vac_wl, model_flux, c="tab:pink", label="Model")
axes[0].legend()

from specutils.utils.wcs_utils import air_to_vac
from astropy import units as u

axes[1].set_title("scheme='inversion' with small y-offsets to see methods")

for i, method in enumerate("Griesen2006, Edlen1953, Edlen1966, Morton2000, PeckReeder1972, Ciddor1996".split(", ")):
    x = air_to_vac(spectrum.λ << u.Angstrom, scheme="inversion", method=method)
    axes[1].plot(x, spectrum.flux / continuum + i * 0.02, label=method)   
    print(method, np.mean(x))
axes[1].legend(ncols=6, fontsize=8)

axes[1].plot(model_vac_wl, model_flux, c="tab:pink")

axes[2].set_title("scheme='iteration' with small y-offsets to see methods")

for i, method in enumerate("Griesen2006, Edlen1953, Edlen1966, Morton2000, PeckReeder1972, Ciddor1996".split(", ")):
    x = air_to_vac(spectrum.λ << u.Angstrom, scheme="iteration", method=method)
    axes[2].plot(x, spectrum.flux / continuum + i * 0.02, label=method)

axes[2].plot(model_vac_wl, model_flux, c="tab:pink")

axes[3].set_title("scheme='Piskunov'")
axes[3].plot(air_to_vac(spectrum.λ << u.Angstrom, scheme='Piskunov'), spectrum.flux / continuum, c='k')
axes[3].plot(model_vac_wl, model_flux, c="tab:pink")




def air_to_vac_greisen2006_phase_refractivity(wl):
    inv_sigma2 = (1 / (wl << u.Angstrom).to(u.um).value)**2
    refractive_index = 1 + 1e-6 * (287.6155 - 1.62887*inv_sigma2 - 0.01360*inv_sigma2**2)
    return wl * refractive_index


def air_to_vac_greisen2006_group_refractivity(wl):
    inv_sigma2 = (1 / (wl << u.Angstrom).to(u.um).value)**2
    refractive_index = 1 + 1e-6 * (287.6155 + 4.88660*inv_sigma2 + 0.06800*inv_sigma2**2)
    return wl * refractive_index


fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)

axes.plot(model_vac_wl, model_flux, c="tab:pink", label="Model")
axes.plot(air_to_vac_greisen2006_phase_refractivity(spectrum.λ), spectrum.flux / continuum, c='tab:blue', label="Phase refractivity")
#axes.plot(air_to_vac_greisen2006_group_refractivity(spectrum.λ), spectrum.flux / continuum, c='tab:orange', label="Group refractivity")
axes.plot(air_to_vac(spectrum.λ << u.Angstrom, scheme="Piskunov"), spectrum.flux / continuum, c="tab:green", ls=":")

