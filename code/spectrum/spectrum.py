
import numpy as np
from typing import Sequence, Optional, Union
from astropy.nddata import NDIOMixin
from astropy import units as u
from functools import cached_property
from specutils.utils.wcs_utils import air_to_vac

from .utils import apply_relativistic_velocity_shift

class Spectrum(NDIOMixin):
    
    def __init__(
        self,
        λ: Sequence[float],
        flux: Sequence[float],
        ivar: Optional[Sequence[float]] = None,
        pixel_flags: Optional[Sequence[int]] = None,
        mask: Optional[Sequence[bool]] = None,
        meta: Optional[dict] = None,
        v_rel: float = 0.0,
        vacuum: Optional[bool] = False,
        continuum: Optional[Union[float, Sequence[float]]] = 1 # should this allow a continuum model of some sort?
    ):
        if ivar is None:
            ivar = np.ones_like(flux)        
        meta = {} if meta is None else meta        
        self.λ = np.array(λ)
        self.flux = np.array(flux)
        self.ivar = np.array(ivar)
        self.mask = np.zeros_like(self.flux, dtype=bool) if mask is None else np.array(mask, dtype=bool)
        self.pixel_flags = pixel_flags
        bad_pixel = ~np.isfinite(self.flux) | ~np.isfinite(self.ivar)
        self.ivar[bad_pixel] = 0        
        self.meta = meta
        self.vacuum = vacuum
        self._v_rel = v_rel
        self.continuum = continuum
        return None
    
    @property
    def v_rel(self):
        return self._v_rel
    
    @v_rel.setter
    def v_rel(self, v_rel):
        self._v_rel = v_rel
                
    @property
    def λ_rest(self):
        return apply_relativistic_velocity_shift(self.λ, -self.v_rel)
    
    @property
    def λ_rest_vacuum(self):
        return apply_relativistic_velocity_shift(self.λ_vacuum, -self.v_rel)
        
    @cached_property
    def λ_vacuum(self):
        if self.vacuum:
            return self.λ
        else:
            # Note: The default method used in specutils is horribly inaccurate! Use Piskunov instead.
            return air_to_vac(self.λ << u.Angstrom, scheme="Piskunov").value


    @property
    def rectified_flux(self):
        return self.flux / self.continuum

    @property
    def rectified_ivar(self):
        return self.ivar * self.continuum**2
    
    def __len__(self):
        return self.λ.size

    def apply_velocity_shift(self, v):
        self.λ = apply_relativistic_velocity_shift(self.λ, v)
        return None


    def plot(self, ax=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
            
        ax.plot(self.λ, self.flux, c='k')
        return fig
        

class SpectrumCollection(Spectrum):

    def get_order(self, index):
        return Spectrum(
            λ=self.λ[index],
            flux=self.flux[index],
            ivar=self.ivar[index],
            v_rel=self.v_rel,
            meta=self.meta,
        )
        

