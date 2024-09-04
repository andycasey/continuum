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
results_2deg = []
results_3deg = []
for i, row in enumerate(tqdm(meta_subset)):

    try:
        path = "../applications/harps/alfCenA/" + row["path"]
        spectra = load_harps_e2ds(path)

        continuum_2deg = []
        continuum_3deg = []
        for spectrum in spectra:
            coeff_2deg = np.polyfit(spectrum.λ, spectrum.flux, 2)
            continuum_2deg.append(np.polyval(coeff_2deg, spectrum.λ))
            coeff_3deg = np.polyfit(spectrum.λ, spectrum.flux, 3)
            continuum_3deg.append(np.polyval(coeff_3deg, spectrum.λ))

        norm_flux_2deg = np.array([s.flux / c for s, c in zip(spectra, continuum_2deg)]).flatten()
        norm_flux_3deg = np.array([s.flux / c for s, c in zip(spectra, continuum_3deg)]).flatten()
        percentiles_2deg = np.percentile(norm_flux_2deg, [90, 95, 97.5])
        percentiles_3deg = np.percentile(norm_flux_3deg, [90, 95, 97.5])

    except:
        print(f"Failured on {row['path']}")

    else:
        results_2deg.append((row["path"], row["snr"], np.nan , np.nan, np.nan, np.std(norm_flux_2deg), *percentiles_2deg))
        results_3deg.append((row["path"], row["snr"], np.nan , np.nan, np.nan, np.std(norm_flux_3deg), *percentiles_3deg))

    '''
    if i > 0:
        import matplotlib.pyplot  as plt
        fig, ax = plt.subplots()
        for s, c in zip(spectra, continuum_2deg):
            ax.plot(s.λ, s.flux / c, c='k')
        raise a
    '''

from astropy.table import Table
t = Table(rows=results_2deg, names=("path", "snr", "status", "cost", "optimality", "std", "p_90", "p_95", "p_97.5"))
t.write("alfCenA_stability_polynomial_2deg.csv", overwrite=True)

t = Table(rows=results_3deg, names=("path", "snr", "status", "cost", "optimality", "std", "p_90", "p_95", "p_97.5"))
t.write("alfCenA_stability_polynomial_3deg.csv", overwrite=True)
