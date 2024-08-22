from glob import glob
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table

from spectrum import Spectrum
from model import apply_radial_velocity_shift


def load_harps_e2ds(path, sign, overall_sign, rest_frame=True, mask_na_doublet=True, ignore_last_order=True):

    pattern = path.split("_e2ds_")[0] + "*_bis*.fits"
    bis_path = glob(pattern)[0]
    with fits.open(bis_path) as image:
        bis_rv = image[0].header["ESO DRS BIS RV"]
        berv = image[0].header["ESO DRS BERV"]

    pattern = path.split("_e2ds_")[0] + "*_ccf_*.fits"
    ccf_path = glob(pattern)[0]
    with fits.open(ccf_path) as image:
        ccf_rv = image[0].header['ESO DRS CCF RV']

    print(bis_rv, ccf_rv, berv)

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
                λ_order = apply_radial_velocity_shift(λ[i], overall_sign * (bis_rv + sign * berv))
            else:
                λ_order = λ[i]

            if mask_na_doublet:
                na_double = (5898 >= λ[i]) * (λ[i] >= 5886)
                ivar[na_double] = 0

            spectra.append(Spectrum(λ_order, flux, ivar, vacuum=False))
            n_pixels += flux.size

    return (spectra, bis_rv, berv)


def vac_to_air(λ):
    return λ / (1 + 2.735182*10**-4 + 131.4182/λ**2 + (2.76249*10**8)/λ**4)

paths = [
    glob("../applications/harps/alfCenA/calib/2011-02-*/*_e2ds_*.fits")[0],
    glob("../applications/harps/alfCenA/calib/2011-03-*/*_e2ds_*.fits")[0],
    glob("../applications/harps/alfCenA/calib/2011-04-*/*_e2ds_*.fits")[0],
    glob("../applications/harps/alfCenA/calib/2011-05-*/*_e2ds_*.fits")[0],
    glob("../applications/harps/alfCenA/calib/2011-06-*/*_e2ds_*.fits")[0],
    glob("../applications/harps/alfCenA/calib/2011-07-*/*_e2ds_*.fits")[0],
]

index = 43

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for path in paths:
    spectra, bis_rv, berv = load_harps_e2ds(path, sign=+1, overall_sign=+1)
    axes[0,0].plot(spectra[index].λ, spectra[index].flux / np.nanmedian(spectra[index].flux), label=path.split("/")[-2])
    spectra, bis_rv, berv = load_harps_e2ds(path, sign=-1, overall_sign=+1)
    axes[1, 0].plot(spectra[index].λ, spectra[index].flux / np.nanmedian(spectra[index].flux), label=path.split("/")[-2])

    spectra, bis_rv, berv = load_harps_e2ds(path, sign=+1, overall_sign=-1)
    axes[0, 1].plot(spectra[index].λ, spectra[index].flux / np.nanmedian(spectra[index].flux), label=path.split("/")[-2])
    spectra, bis_rv, berv = load_harps_e2ds(path, sign=-1, overall_sign=-1)
    axes[1, 1].plot(spectra[index].λ, spectra[index].flux / np.nanmedian(spectra[index].flux), label=path.split("/")[-2])

#import pickle
#with open("../applications/harps-sandbox/20240816_train_harps_model.pkl", "rb") as fp:
#    λ, label_names, parameters, W, H = pickle.load(fp)

#si, ei = λ.searchsorted(axes[0,0].get_xlim())
#for ax in axes.flat:
#    ax.plot(λ[si:ei], np.exp(-H[16, si:ei]), c="tab:blue")


import bz2
def read_spectrum(path):
    if path.endswith(".asc"):
        wl, flux, continuum = np.loadtxt(path).T
    elif path.endswith(".bz2"):
        fp = bz2.BZ2File(path)
        data = np.array(fp.read().decode().split(), dtype=float)
        data = data.reshape((-1, 3))
        wl, flux, continuum = data.T
    return wl, (flux / continuum)

wl, norm_flux = read_spectrum("amp00cp00op00t5750g45v20modrt0b300000rs.asc.bz2")
si, ei = wl.searchsorted(axes[0,0].get_xlim())
for ax in axes.flat:
    ax.plot(vac_to_air(wl[si:ei]), norm_flux[si:ei], c="tab:orange")

import h5py as h5
with h5.File("korg_synth_alfCen", "r") as fp:
    for ax in axes.flat:
        ax.plot(vac_to_air(fp["wl"][:]), fp["norm_flux"][:], c="tab:red")
#si, ei = λ.searchsorted(axes[0,0].get_xlim())
#for ax in axes.flat:
#    ax.plot(air_to_vacuum(λ[si:ei]), np.exp(-H[23, si:ei]), c="tab:red")

axes[0, 0].set_title("shift by bis_rv + berv")
axes[1, 0].set_title("shift by bis_rv - berv")
axes[0, 1].set_title("shift by -(bis_rv + berv)")
axes[1, 1].set_title("shift by -(bis_rv - berv)")

fig.savefig("alfCen_check_bis_berv.png")

from astropy import units as u
from specutils.utils.wcs_utils import air_to_vac


# Now show all in vacuum
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)

schemes = ("inversion", "Piskunov", "iteration")
for (ax, scheme) in zip(axes, schemes):
    
    for path in paths:
        spectra, bis_rv, berv = load_harps_e2ds(path, sign=-1, overall_sign=-1)
        x = air_to_vac(spectra[index].λ << u.Angstrom, scheme=scheme)
        y = spectra[index].flux / np.nanmedian(spectra[index].flux)
        coeff = np.polyfit(x, y, 2)
        cont = np.polyval(coeff, x.value)

        ax.plot(x, y / cont, label=path.split("/")[-2])

    si, ei = wl.searchsorted(ax.get_xlim())
    ax.plot(wl[si:ei], norm_flux[si:ei], c="tab:orange")
    ax.set_title(scheme)

