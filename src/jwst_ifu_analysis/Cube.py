import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.constants import c
import photutils.aperture as ap
from regions import Regions

from .Spectrum import Spectrum
from .utils import *


class Cube:
    """A class representing a spectral data cube with methods for extraction, manipulation, and I/O."""
    def __init__(self, data, header=None):
        """Initialize a Cube object.

        Args:
            data (str or np.ndarray): FITS filename or data array.
            header (astropy.io.fits.Header, optional): FITS header. Required if data is an array.
        """

        # if data is a file, load in the data and header
        if isinstance(data, str):
            self.data, self.header = fits.getdata(data, "SCI", header=True)

        else:
            self.data = data
            self.header = header

        # get wvl axis and wcs info
        self.wvl_axis = get_wvl_axis(self.header)
        self.wcs = WCS(self.header)
        self.shape = np.shape(self.data)

        # Convert data to Jy if necessary
        if self.header["BUNIT"] == "MJy/sr":
            pix_area = self.header["PIXAR_SR"]
            self.data *= 1.0e6 * pix_area
            self.header["BUNIT"] = "Jy"

    def extract_spectrum(self, region):
        """Extract a 1D spectrum from an IFU cube

        Args:
            region (str or region.Regions): Region file defining the aperture used to extract the spectrum

        Returns:
            (Spectrum): Extracted 1D spectrum
        """

        nchan = self.shape[0]

        # load in the region and convert to an aperture
        if isinstance(region, str):
            reg = Regions.read(region)[0]
        else:
            reg = region
        sky_ap = ap.region_to_aperture(reg)

        # convert sky aperture to a pixel aperture
        pix_ap = sky_ap.to_pixel(self.wcs.celestial)

        # convert aperture to a mask
        mask = pix_ap.to_mask(method="exact")

        # initialize spectrum array
        spectrum_flux = np.zeros(nchan)
        # extract the 1D spectrum
        for i in range(len(spectrum_flux)):
            # get data of current channel
            chan = self.data[i]
            # extract the data within the aperture
            ap_data = mask.get_values(chan)

            # if all data is nans, set spectrum flux to nan
            if np.isnan(ap_data).all():
                spectrum_flux[i] = np.nan
            else:
                # sum to get value for spectrum
                spectrum_flux[i] = np.nansum(ap_data)

        return Spectrum(wvl_axis=self.wvl_axis, flux=spectrum_flux)

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

        return Cube(data=region_data, header=header)

    def cont_fit(self, line_list):
        """Continuum fit the spectral cube

        Args:
            line_list (list of dict): list of lines used to mask out lines when fitting the continuum

        Returns:
            cont_cube (Cube): Fitted continuum Cube
        """

        # create holder arrays for continuum and continuum
        # subtracted cubes
        cont_data = np.zeros(self.data.shape)
        cont_sub_data = np.zeros(self.data.shape)

        # perform the continuum subtraction for each pixel
        nchan, ni, nj = self.shape
        for i in range(ni):
            for j in range(nj):
                # fit the continuum to i, j spectrum
                pixel_spectrum = Spectrum(
                    wvl_axis=self.wvl_axis, flux=self.data[:, i, j]
                )
                continuum = pixel_spectrum.fit_continuum(line_list)
                # set the i, j spectrum for the cont and cont sub cubes
                cont_data[:, i, j] = continuum.flux

        # create continuum and continuum subtracted cube objects
        cont_cube = Cube(data=cont_data, header=self.header)

        return cont_cube

    def write(self, filename):
        """Write the cube data and header to a FITS file.

        Args:
            filename (str): Output FITS filename.
        """
        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        hdu.writeto(filename, overwrite=True)


def combine_cubes(old_cube, new_cube):
    """Align and combine two Cube objects into a single Cube.

    Args:
        old_cube (Cube): First cube object (reference).
        new_cube (Cube): Second cube object to align and combine.

    Returns:
        Cube: Combined cube object with updated header and data.
    """
    # take median to get representative image
    img1 = np.nan_to_num(np.nanmedian(old_cube.data, axis=0))
    img2 = np.nan_to_num(np.nanmedian(new_cube.data, axis=0))

    # take cross correlation to find offset
    corr = correlate2d(img1, img2, mode="full", boundary="fill", fillvalue=0)
    # find the peak of the correlation
    y_max, x_max = np.unravel_index(np.argmax(corr), corr.shape)
    # calculate the offset
    y_offset = (img1.shape[0] - 1) - y_max
    x_offset = (img1.shape[1]) - x_max

    # align two images and use entire FOV
    # get dimensions of original image
    ny, nx = img1.shape
    # get dimensions of final image
    nx_big = np.abs(x_offset) + nx
    ny_big = np.abs(y_offset) + ny

    # determine where to place the new/ old arrays based on the direction of
    # the offset, also how much to shift the reference pixels
    if x_offset < 0 and y_offset < 0:
        old_x_bounds = (0, nx)
        old_y_bounds = (0, ny)
        new_x_bounds = (np.abs(x_offset), nx_big)
        new_y_bounds = (np.abs(y_offset), ny_big)
        x_shift, y_shift = -x_offset, -y_offset
    elif x_offset < 0 and y_offset > 0:
        old_x_bounds = (0, nx)
        old_y_bounds = (np.abs(y_offset), ny_big)
        new_x_bounds = (np.abs(x_offset), nx_big)
        new_y_bounds = (0, ny)
        x_shift, y_shift = -x_offset, 0
    elif x_offset > 0 and y_offset < 0:
        old_x_bounds = (np.abs(x_offset), nx_big)
        old_y_bounds = (0, ny)
        new_x_bounds = (0, nx)
        new_y_bounds = (np.abs(y_offset), ny_big)
        x_shift, y_shift = 0, -y_offset
    elif x_offset > 0 and y_offset > 0:
        old_x_bounds = (np.abs(x_offset), nx_big)
        old_y_bounds = (np.abs(y_offset), ny_big)
        new_x_bounds = (0, nx)
        new_y_bounds = (0, ny)
        x_shift, y_shift = 0, 0

    # get bounds of overlap region
    x1 = np.abs(x_offset) + 1
    x2 = -np.abs(x_offset) - 1
    y1 = np.abs(y_offset) + 1
    y2 = -np.abs(y_offset) - 1

    # create footprint for the combined cube
    combined_cube = np.zeros((old_cube.data.shape[0], ny_big, nx_big))

    # shift each channel
    for i in range(old_cube.data.shape[0]):
        # get new and old channel
        old_chan = np.nan_to_num(old_cube.data[i])
        new_chan = np.nan_to_num(new_cube.data[i])

        big_chan = np.zeros((ny_big, nx_big))
        # add the old channel
        big_chan[
            old_y_bounds[0] : old_y_bounds[1], old_x_bounds[0] : old_x_bounds[1]
        ] += old_chan
        # add the new channel
        big_chan[
            new_y_bounds[0] : new_y_bounds[1], new_x_bounds[0] : new_x_bounds[1]
        ] += new_chan
        # divide overlap region by 2 to not double count (average)
        big_chan[y1:y2, x1:x2] /= 2
        # set 0s to nans
        big_chan[big_chan == 0] = np.nan
        combined_cube[i] = big_chan

    # update the header
    new_header = new_cube.header
    combined_header = new_header.copy()
    # get reference pixel position
    ref_1 = new_header["CRPIX1"]
    ref_2 = new_header["CRPIX2"]
    # set new dimensions
    combined_header["NAXIS1"] = ny_big
    combined_header["NAXIS2"] = nx_big
    # set new reference pixel position
    combined_header["CRPIX1"] = ref_1 + x_shift
    combined_header["CRPIX2"] = ref_2 + y_shift

    # create a return a new cube object
    return Cube(data=combined_cube, header=combined_header)
