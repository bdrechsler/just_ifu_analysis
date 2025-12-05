import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.io import fits
from astropy.modeling import fitting, models
from lmfit.models import GaussianModel
from astropy.table import Table
from scipy.ndimage import median_filter
import emcee

from .utils import get_line_mask


class Spectrum:
    """Class to handle 1D spectra

    Args:
        wvl_axis (np.array): Wavelength axis in microns
        flux (np.array): Flux axis in Jy
    """

    def __init__(self, wvl_axis, flux):
        self.wvl_axis = wvl_axis
        self.flux = flux

    def __add__(self, other):
        # if (self.wvl_axis != other.wvl_axis).any():
        if not np.array_equal(self.wvl_axis, other.wvl_axis, equal_nan=True):
            print("Spectral axis are not matched, cannot add.")
            return
        else:
            summed_flux = self.flux + other.flux
            return Spectrum(wvl_axis=self.wvl_axis, flux=summed_flux)

    def __sub__(self, other):
    # if (self.wvl_axis != other.wvl_axis).any():
        if not np.array_equal(self.wvl_axis, other.wvl_axis, equal_nan=True):
            print("Spectral axis are not matched, cannot add.")
            return
        else:
            sub_flux = self.flux - other.flux
            return Spectrum(wvl_axis=self.wvl_axis, flux=sub_flux)

    def fit_continuum(self, line_list, med_kernel=3, fit_order=3):
        """Fit the continuum of the spectrum, ignore line emission

        Args:
            line_list (list): list of lines, will mask out any in the wavelength range
            med_kernel (odd int): Size of kernel used in
            median smoothing. Defaults to 3.
            fit_order (int): Order of polynomial to fit to the continuum. Defaults to 3.

        Returns:
            continuum_spectrum (Spectrum): Spectrum of the fitted continuum
        """

        # create a mask for all nearby lines
        cont_mask = get_line_mask(line_list=line_list, wvl_axis=self.wvl_axis)

        # mask the wavelength and flux
        wvl_axis_masked = self.wvl_axis[cont_mask]
        flux_masked = self.flux[cont_mask]

        # median smooth then fit a polynomial to the continuum spectrum
        cont_flux_smooth = median_filter(
            flux_masked.astype(np.float64), size=med_kernel
        )
        fit_params = np.polyfit(wvl_axis_masked, cont_flux_smooth, fit_order)
        model_cont_flux = np.poly1d(fit_params)

        return Spectrum(wvl_axis=self.wvl_axis, flux=model_cont_flux(self.wvl_axis))

    def spectral_region(self, center_wvl, region_width):
        """Extract a spectral region around a center wavelength
        Args:
            center_wvl (float): Center wavelength in microns
            region_width (float): Width of the region in microns
        Returns:
            region_spectrum (Spectrum): spectral region"""

        wvl_min = center_wvl - (region_width / 2.0)
        wvl_max = center_wvl + (region_width / 2.0)

        region_mask = (self.wvl_axis > wvl_min) & (self.wvl_axis < wvl_max)

        region_flux = self.flux[region_mask]
        region_wvl = self.wvl_axis[region_mask]

        return Spectrum(wvl_axis=region_wvl, flux=region_flux)

    def fit_gaussian(self, line):
        """Fit a Gaussian to a spectral line

        Args:
            line (dict): dictionary with line information, must contain
                         "rest_wvl" and "line_width" keys
        Returns:
            fit_result (dict): dictionary with fit results and uncertainties
        """

        # mask out nan channels
        wvl = self.wvl_axis[~np.isnan(self.wvl_axis)]
        flux = self.flux[~np.isnan(self.wvl_axis)]

        # initial guess of parameters
        amp = np.max(flux)
        mean = line["rest_wvl"]
        stddev = line["line_width"] / 3.
        g_init = models.Gaussian1D(amplitude=amp, mean=mean, stddev=stddev)
        # fit the gaussian to the data
        fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
        sigma = self.estimate_sigma(line)
        weights = [1 / sigma] * len(wvl)

        g = fitter(g_init, wvl, flux, weights=weights)
        info = fitter.fit_info
        cov = info['param_cov']
        errs = np.sqrt(np.diag(cov))

        fit_result = {"amp": (g.amplitude.value, errs[0]),
                          "mean": (g.mean.value, errs[1]),
                          "std": (g.stddev.value, errs[2])}

        return fit_result

    def line_flux(self, line, method="fit", return_fit=False):
        """Calculate the line flux of a spectral line

        Args:
            line (dict): dictionary with line information, must contain
                         "rest_wvl" and "line_width" keys
            method (str): method to calculate line flux, either "fit" or "integral"
            return_fit (bool): whether to return the fit result when using "fit" method
        Returns:
            line_flux (float): line flux in erg/s/cm^2
            line_flux_err (float): uncertainty in line flux (only for "fit" method)
            fit_result (dict): dictionary with fit results and uncertainties (only for "fit" method and if return_fit is True)
        """

        if method == "fit":
            # fit gaussian
            fit_result = self.fit_gaussian(line)
            amp, amp_err = fit_result['amp']
            sigma_wvl, sigma_wvl_err = fit_result['std']

            # switch sigma to nu binning
            sigma_nu = (sigma_wvl * u.um * c / (line["rest_wvl"] * u.um) ** 2).to(u.Hz)
            sigma_nu_err = (sigma_wvl_err * u.um * c / (line["rest_wvl"] * u.um) ** 2).to(u.Hz)

            # calculate flux as area under gaussian
            line_flux_fit = amp * u.Jy * sigma_nu * np.sqrt(2 * np.pi)
            line_flux_fit = line_flux_fit.to(u.erg / u.s / u.cm**2).value

            line_flux_err = line_flux_fit * np.sqrt((amp_err/amp)**2 + (sigma_nu_err / sigma_nu)**2)
            if return_fit:
                return line_flux_fit, line_flux_err, fit_result
            else:
                return line_flux_fit, line_flux_err

        elif method == "integral":
            # calculate line flux as numerical integral
            line_region = self.spectral_region(
                center_wvl=line["rest_wvl"], region_width=line["line_width"]
            )
            # get dlam and convert to dnu
            dlam = line_region.wvl_axis[1] - line_region.wvl_axis[0]
            dnu = (dlam * u.um * c / (line["rest_wvl"] * u.um) ** 2).to(u.Hz)
            # estimate the integral
            line_flux_integral = np.nansum(line_region.flux) * u.Jy * dnu
            line_flux_integral = line_flux_integral.to(u.erg / u.s / u.cm**2).value

            return line_flux_integral

    def get_vel_axis(self, line):
        """Get velocity axis given a spectral line

        Args:
            line (dict): dictionary with line information, must contain
                         "rest_wvl" key
        Returns:
            vel_axis (np.array): velocity axis in km/s
        """

        # get velocity axis
        dlam = (self.wvl_axis - line["rest_wvl"]) * u.um
        rest_wvl = line["rest_wvl"] * u.um
        return (c * (dlam / rest_wvl)).to(u.km / u.s).value

    def estimate_sigma(self, line):
        """Estimate the noise in the spectrum using line-free regions

        Args:
            line (dict): dictionary with line information, must contain
                         "rest_wvl" and "line_width" keys
        Returns:
            sigma (float): estimated noise in the spectrum
        """

        # get line free region of spectrum
        l_left = line['rest_wvl'] - line['line_width']
        l_right = line['rest_wvl'] + line['line_width']

        line_free = self.flux[(self.wvl_axis < l_left) |
                              (self.wvl_axis > l_right)]

        mad = np.nanmedian(np.abs(line_free - np.nanmedian(line_free)))
        #return np.std(line_free)
        return mad * 1.5

    def write(self, fname):
        """Write out spectrum to file

        Args:
            fname (str): path to output file
        """

        # save as astropy table
        t = Table([self.wvl_axis, self.flux], names=("wvl_axis", "flux"))

        t.write(fname, overwrite=True)

    @classmethod
    def read(cls, fname):
        """Read in file to create cube object

        Args:
            fname (str): path to input file
        """

        # get the data and header
        t = Table.read(fname)

        return cls(wvl_axis=t["wvl_axis"], flux=t["flux"])

