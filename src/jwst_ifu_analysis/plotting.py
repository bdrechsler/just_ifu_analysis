import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def recenter(wcs, ax, x=69.9744522, y=26.0526275, r=0.000595):
    """recenter the plot around a given sky position

    Args:
        wcs (WCS): wcs object of the image
        ax (axes): axes object of the plot
        x (float): ra of the center position in degrees
        y (float): dec of the center position in degrees
        r (float): radius around the center position in degrees
    """

    wcs = wcs.celestial
    xpix, ypix = wcs.wcs_world2pix(x, y, 0)
    pix_scale = proj_plane_pixel_scales(wcs)
    sx, sy = pix_scale[1], pix_scale[0]  # get the pixel scale for each dimension

    dx_pix = r / sx  # get number of pixels in each direction
    dy_pix = r / sy

    ax.set_xlim(xpix - dx_pix, xpix + dx_pix)
    ax.set_ylim(ypix - dy_pix, ypix + dy_pix)


def scale_bar(file, ax, color="k", start_pos=(0.6, 0.1), label=True, fontsize=12):
    """add a scale bar to the plot

    Args:
        file (str): path to the fits file
        ax (axes): axes object of the plot
        color (str): color of the scale bar
        start_pos (tuple): position of the scale bar in axes coordinates
        label (bool): whether to add a label to the scale bar
        fontsize (int): fontsize of the label"""

    header = fits.getheader(file, "SCI")
    # get au per pixel for scale bar
    deg_per_pixel = header["CDELT1"]
    arcsec_per_pix = (deg_per_pixel * u.deg).to(u.arcsec)
    au_per_pix = arcsec_per_pix.value * 140.0  # multiply by distance in pc
    # get number of pixels in 100 au
    pix_in_100au = 100 / au_per_pix

    # transform from axes to data units
    inv = ax.transLimits.inverted()
    # starting position of scalebar
    scalex1, scaley1 = inv.transform(start_pos)
    # ending position of scalebar
    scalex2 = scalex1 + pix_in_100au

    if label:
        # add text to scale bar
        txt = ax.text(
            start_pos[0],
            start_pos[1] + 0.03,
            "100 au",
            transform=ax.transAxes,
            color=color,
            fontsize=fontsize,
            #backgroundcolor='black'
        )
        #txt.set_bbox({"alpha": 1})

    # plot the scale bar
    scale_x = np.linspace(scalex1, scalex2, 10)
    scale_y = [scaley1] * len(scale_x)
    ax.plot(scale_x, scale_y, color=color)


def format_ticks(ax, nrows, row_ind, col_ind):
    """format the tick labels of a subplot in a grid of subplots

    Args:
        ax (axes): axes object of the subplot
        nrows (int): number of rows in the grid of subplots
        row_ind (int): row index of the subplot
        col_ind (int): column index of the subplot
    """

    # get ra and dec
    ra = ax.coords[0]
    dec = ax.coords[1]

    # set where tick labels should be shown
    if row_ind == nrows - 1:
        ra.set_axislabel("RA (ICRS)", minpad=0.5, fontsize=10)
    else:
        ra.set_ticklabel_visible(False)
    if col_ind == 0:
        dec.set_axislabel("DEC (ICRS)", minpad=-0.5, fontsize=10)
    else:
        dec.set_ticklabel_visible(False)

    # set ticklabel fontsize and number of ticks
    ra.set_ticklabel(fontsize=8)
    dec.set_ticklabel(fontsize=8)
    ra.set_ticks(number=4)
    dec.set_ticks(number=4)

def plot_star_pos(wcs, ax, star_pos):
    """plot the position of the star on the image

    Args:
        wcs (WCS): wcs object of the image
        ax (axes): axes object of the plot
        star_pos (SkyCoord): position of the star
    """

    pix_pos = star_pos.to_pixel(wcs)
    ax.plot(pix_pos[0], pix_pos[1], marker='*', color='cyan', ms=10)
