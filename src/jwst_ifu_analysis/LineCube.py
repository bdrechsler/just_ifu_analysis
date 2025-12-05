from astropy.constants import c
import astropy.units as u
from astropy.io import fits
import bettermoments as bm

from .Cube import Cube
from .utils import *


class LineCube(Cube):
    def __init__(self, data, line, line_list, header=None):
        """LineCube class for handling spectral cubes centered on a specific line

        Args:
            data (numpy.ndarray or str): 3D array of spectral cube data or path to FITS file
            line (dict): dictionary with line information (e.g., rest wavelength)
            line_list (list): list of lines to consider for masking
            header (astropy.io.fits.Header, optional): FITS header. Defaults to None.
        """

        super().__init__(data, header)

        self.line = line
        self.line_list = line_list

        # get a mask for the lines in this cube
        mask = get_line_mask(
            line_list=line_list, wvl_axis=self.wvl_axis, central_line=self.line
        )

        # replace masked channels with nans
        for i in range(len(mask)):
            if not mask[i]:
                self.data[i, :, :] = np.nan
                self.wvl_axis[i] = np.nan

        # get velocity axis
        dlam = (self.wvl_axis - self.line["rest_wvl"]) * u.um
        rest_wvl = self.line["rest_wvl"] * u.um
        self.vel_axis = (c * (dlam / rest_wvl)).to(u.km / u.s).value

    def spectral_region(self, center_wvl, region_width):
        """Get a spectral region (slab) of the spectral cube

        Args:
            center_wvl (float): central wavelength of the region [um]
            region_width (float): width of the spectral region [um]

        Returns:
            spectral_region (Cube): region of spectral cube within spectral
            window
        """
        wvl_min = center_wvl - (region_width / 2.0)
        wvl_max = center_wvl + (region_width / 2.0)

        region_data = self.data[(self.wvl_axis > wvl_min) & (self.wvl_axis < wvl_max)]

        # update header with new wvl range
        header = self.header.copy()
        header["CRVAL3"] = wvl_min
        header["NAXIS3"] = np.shape(region_data)[0]

        return LineCube(data=region_data, line=self.line, line_list=self.line_list, header=header)

    def moment0(self, fname=None, units="Jy"):
        """Create a moment 0 map

        Args:
            fname (str, optional): Path to save map. Defaults to None.

        Returns:
            numpy.ndarray: moment 0 map
        """
        # assume the cube was initiated with a line list
        dv = self.vel_axis[1] - self.vel_axis[0]
        mom0 = np.nansum(self.data, axis=0) * dv

        # replace 0's with nans
        mom0[mom0 == 0] = np.nan

        # optionally convert to mJy
        if units == "mJy":
            mom0 *= 1000

        # make the header 2d header
        header_2d = make_header_2d(self.header)

        # optionally save the map
        if fname:
            hdu = fits.PrimaryHDU(data=mom0, header=header_2d)
            hdu.writeto(fname, overwrite=True)

        return mom0

    def moment1(self, fname=None):
        """Create a moment 1 map
        Args:
            fname (str, optional): Path to save map. Defaults to None.
        Returns:
            numpy.ndarray: moment 1 map"""
        dv = self.vel_axis[1] - self.vel_axis[0]

        vel_cube = np.zeros(self.data.shape)
        for i in range(self.shape[0]):
            vel_cube[i] = np.full((self.shape[1], self.shape[2]), self.vel_axis[i])

        mom0 = self.moment0()

        # use first and last two channels to estimate sigma for a
        # line free channel
        sigmas = [np.nanstd(self.data[i]) for i in range(-2, 2)]
        sigma_chan = np.nanmean(sigmas)
        # error propogation to get error on M0
        sigma_map = np.sqrt(len(self.data)) * sigma_chan * dv
        thresh = 4 * sigma_map

        # # create a mask for the M1 map
        mask = np.ones((self.shape[1], self.shape[2]))
        mask[(mom0 > -thresh) & (mom0 < thresh)] = 0

        mom1 = (np.nansum(vel_cube * self.data, axis=0) * dv) / mom0
        mom1 *= mask

        # replace 0's with nans
        mom1[mom1 == 0] = np.nan

        # make the header 2d header
        header_2d = make_header_2d(self.header)

        # optionally save the map
        if fname:
            hdu = fits.PrimaryHDU(data=mom1, header=header_2d)
            hdu.writeto(fname, overwrite=True)

        return mom1

    def bm_collapse(self, order, smooth=3, clip=3.0):
        """Create moment maps using the bettermoments package

        Args:
            order (int): order of the moment (0 or 1)
            smooth (int, optional): smoothing kernel size. Defaults to 3.
            clip (float, optional): sigma clipping level. Defaults to 3.0.
        Returns:
            numpy.ndarray: moment map
        """

        # smooth the data
        if smooth:
            data = bm.smooth_data(self.data, smooth=smooth)
        else:
            data = self.data
        # estimate the noise
        rms = bm.estimate_RMS(data=data, N=5)
        # get a threshold mask
        if clip:
            mask = bm.get_threshold_mask(data=data, clip=clip)
            data *= mask

        if order == 0:
            mom = bm.collapse_zeroth(velax=self.vel_axis, data=data, rms=rms)

        elif order == 1:
            mom = bm.collapse_first(velax=self.vel_axis, data=data, rms=rms)

        return mom
