import numpy as np
from scipy.signal import correlate2d


def get_wvl_axis(header):
    """Get the wavelength axis from IFU cube header

    Args:
        header (header): IFU fits header object

    Returns:
        numpy.ndarray: Wavelength axis arrray of the IFU cube
    """
    # get wavelength axis
    start_wvl = header["CRVAL3"]
    wvl_step = header["CDELT3"]
    nchan = header["NAXIS3"]
    wvl_axis = (np.arange(nchan) * wvl_step) + start_wvl
    return wvl_axis


def get_line_mask(line_list, wvl_axis, central_line=None):
    """Get a mask for the wavelength axis that filters out spectral lines

    Args:
        line_list (list): List of spectral lines
        wvl_axis (numpy.ndarray): Wavelength axis of the IFU cube
        central_line (dict, optional): Central line to exclude from masking. Defaults to None.
    Returns:
        numpy.ndarray: Boolean mask for the wavelength axis"""

    wvl_min = np.min(wvl_axis)
    wvl_max = np.max(wvl_axis)
    # get list of lines within wvl range
    nearby_lines = [
        l for l in line_list if (l["rest_wvl"] > wvl_min) & (l["rest_wvl"] < wvl_max)
    ]

    if central_line:
        # find the line closest to the middle of the cube
        # remove the line from the list of nearby lines
        nearby_lines = [
            l for l in nearby_lines if l["rest_wvl"] != central_line["rest_wvl"]
        ]

    # create a mask for the lines in the spectrum
    if len(nearby_lines) > 0:
        line_mask = np.zeros(len(wvl_axis), dtype=bool)
        for l in nearby_lines:
            lo = l["rest_wvl"] - (0.5 * l["line_width"])
            hi = l["rest_wvl"] + (0.5 * l["line_width"])
            mask = (wvl_axis > lo) & (wvl_axis < hi)
            line_mask = line_mask | mask
        # invert the line mask since we want to include the continuum
        return ~line_mask
    else:
        # if there are no nearby lines, don't filter anything
        return np.ones(len(wvl_axis), dtype=bool)


def get_chan(wvl):
    """Get the channel name for a given wavelength

    Args:
        wvl (float): Wavelength in microns
    Returns:
        str: Channel name
    """

    chan_bounds = {
        "nirspec": [2.8708948855637573, 5.269494898093398],
        "ch1-long": [6.530400209798245, 7.649600181524875],
        "ch2-long": [10.010650228883605, 11.699350233480798],
        "ch3-long": [15.41124984738417, 17.978749789996073],
        "ch4-long": [24.40299961855635, 28.69899965589866],
        "ch1-short": [4.900400095357327, 5.739600074157352],
        "ch1-medium": [5.6603998474020045, 6.629999822907848],
        "ch2-short": [7.5106502288836055, 8.770350232312921],
        "ch2-medium": [8.670650076295715, 10.13055008027004],
        "ch3-short": [11.551250190706924, 13.47125014779158],
        "ch3-medium": [13.341250152559951, 15.568750102771446],
        "ch4-short": [17.70300076296553, 20.94900079118088],
        "ch4-medium": [20.693000534083694, 24.47900056699291],
    }

    line_chan = 0
    # loop through the channels to see see if input wvl is within its window
    for chan, bounds in chan_bounds.items():
        if wvl > bounds[0] and wvl < bounds[1]:
            line_chan = chan
            break  # if the wvl is in the current channel, set line_chan

    return line_chan


def make_header_2d(header_3d):
    """Make a 2D header from a 3D header by removing the spectral axis

    Args:
        header_3d (header): 3D fits header object
    Returns:
        header: 2D fits header object
    """

    # edit header info
    header = header_3d.copy()
    header["BUNIT"] = "Jy"
    header["NAXIS"] = 2
    header["WCSAXES"] = 2
    del header["NAXIS3"]
    del header["CRPIX3"]
    del header["CDELT3"]
    del header["CRVAL3"]
    del header["CTYPE3"]
    del header["CUNIT3"]
    del header["PC3_1"]
    del header["PC3_2"]
    del header["PC3_3"]
    del header["PC1_3"]
    del header["PC2_3"]
    del header["DISPAXIS"]
    del header["VELOSYS"]
    del header["SPECSYS"]

    return header
