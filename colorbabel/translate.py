"""Quickly translate between different color representations."""
import seaborn.palettes as pl
from seaborn import palplot
from IPython.display import HTML
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import colorlover as cl

__all__ = ['ColorTranslator']


class ColorTranslator(object):
    """
    Quickly translate color information between several types.

    After initializing with a list of colors, they will be converted
    to a matplotlib colormap. They can then either be:

    1. Converted to another type via methods, e.g., `to_numeric`
    2. Subsetted, and then converted to another type, e.g.,
       `self([.1, .3, .5], kind='husl')`

    Parameters
    ----------
    colors : array of rgb floats | array of rgb/rgba strings |
             matplotlib colormap | array-like of hex codes |
             array-like of color names
        The colors you wish to interpolate between. If array
        of floats, must all be either between 0 and 1, or
        0 and 255.

    Attributes
    ----------
    cmap : matplotlib colormap
        The colormap created by blending the colors given in `colors`.
    """
    def __init__(self, colors):
        if isinstance(colors, (LinearSegmentedColormap, ListedColormap)):
            colors = colors
        else:
            if not isinstance(colors[0], str):
                # Consider it an array of rgb/rgba
                colors = np.atleast_2d(colors).astype(float)
                if colors.shape[-1] not in [3, 4]:
                    raise ValueError('if floats/ints, colors must have'
                                     ' a last dimension of shape 3 or 4')
                if not ((colors >= 0) * (colors <= 1)).all():
                    colors = colors / 255.
            else:
                if 'rgb' in colors[0]:
                    # Convert strings to numeric first
                    colors = np.array(cl.to_numeric(colors)) / 255.
                elif '#' in colors[0]:
                    # Assume it's hex and we can just pass
                    pass
                else:
                    # Try to convert with webcolors
                    colors = _names_to_rgb(colors)
            colors = pl.blend_palette(colors, as_cmap=True)
        self.cmap = colors

    def to_numeric(self, n_bins=255, kind='rgb'):
        """Convert colormap to numeric array.

        Parameters
        ----------
        n_bins : int
            The number of bins in the output array
        kind : 'rgb' | 'husl'
            Whether to return colors in rgb or husl format

        Returns
        -------
        colors_numeric : array, shape (n_bins, 4)
            An array of rgb / husl colors, evenly spaced through
            self.colormap.
        """
        colors_numeric = self.cmap(np.linspace(0, 1, n_bins))
        if kind == 'husl':
            colors_numeric = np.array([pl.husl.rgb_to_husl(r, g, b)
                                       for r, g, b, _ in colors_numeric])
        elif kind == 'rgb':
            pass
        else:
            raise ValueError("kind {} not supported".format(kind))
        return colors_numeric

    def to_strings(self, n_bins=255, kind='rgb'):
        """Convert colormap to plotly-style strings.

        Parameters
        ----------
        n_bins : int
            The number of bins in the output array
        kind : 'rgb' | 'husl' | 'hex' | 'name'
            Whether to return colors in rgb, husl, or hex format.
            Or, return the common name for that color if possible.

        Returns
        -------
        colors_string : list of strings
            A list of "rgb()" or "hsl()" strings suitable for
            use in plotly. Or a list of hex strings for online plotting. Or
            a list of names associated with the colors.
        """
        # Remove the alpha
        array = self.cmap(np.linspace(0, 1, n_bins))[:, :-1]
        if kind == 'hex':
            colors_string = [pl.husl.rgb_to_hex(ii) for ii in array]
        elif kind == 'name':
            colors_string = _get_color_names(array * 255.)
        else:
            array = array * 255.
            list_of_tups = [tuple(i) for i in array]
            if kind == 'rgb':
                colors_string = cl.to_rgb(list_of_tups)
            elif kind == 'husl':
                colors_string = cl.to_hsl(list_of_tups)
        return colors_string

    def to_diverging(self, center='light', mid_spread=.4, log_amt=1e-3,
                     as_cmap=True, **kwargs):
        """Convert diverging colormap from first/last color in self.

        Parameters
        ----------
        center : array, string
            The color of the new center value for the cmap. If a string,
            must be one of ['light', 'dark']. If an array, must be RGB
            of length 3, values between 0 and 1.
        mid_spread : float between 0 and 1
            The amount of spread for the middle color. Values closer to 1
            will cause the middle color to take up a larger part of the
            resulting colormap.
        log_amt : float
            The middle color drops off logarithmically. This defines how
            quickly this happens. Larger numbers cause this to drop faster
            and 1 means that the dropoff is immediate. Reccomended between
            1e-3 and 1e-1.
        as_cmap : bool
            Whether to return the diverging map as a list of colors
            or a LinearSegmentedColormap

        Returns
        -------
        colors_div : list of arrays | matplotlib colormap
            The diverging colormap, created by interpolating between
            the first and last colors in self.cmap.
        """
        out = _add_middle_color(self.to_numeric(), midpoint=center,
                                mid_spread=mid_spread, log_amt=log_amt)
        return pl.blend_palette(out, as_cmap=as_cmap)

    def show_colors(self, n_bins=None):
        """Display the colors in self.cmap.

        Parameters
        ----------
        n_bins : None | int
            How many bins to use in displaying the colormap. If None,
            a continuous representation is used and plotted with
            matplotlib.
        """
        if isinstance(n_bins, int):
            palplot(self.to_numeric(n_bins))
        elif n_bins is None:
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            f, ax = plt.subplots(figsize=(5, .5))
            ax.imshow(gradient, aspect='auto', cmap=self.cmap)
            ax.set_axis_off()
        else:
            raise ValueError('n_bins must be type int or None')

    def __call__(self, data, kind='rgb', as_string=False,
                 vmin=None, vmax=None):
        """Convert a subset of colors to a given type.

        Parameters
        ----------
        data : array, shape (n_colors,)
            Must be an array of floats between 0 and 1. These
            will be used to index into self.cmap.
        kind : 'rgb' | 'husl' | 'html'
            Whether to return output colors in rgb or husl space.
            If 'html', the color output of the call will be displayed.
        as_string : bool
            If True, return colors as plotly-style strings.

        Returns
        -------
        arr : np.array | list of strings
            The colors associated with values in `frac`.
        """
        data = np.atleast_1d(data)
        if data.ndim > 1:
            raise ValueError('frac must be 1-d')
        if vmin is not None or vmax is not None:
            # If we need to scale out data
            frac = np.clip((data - vmin) / float(vmax - vmin), 0, 1)
        else:
            frac = data
        if not ((frac >= 0.) * (frac <= 1.)).all(0):
            raise ValueError('input must be between 0 and 1, you'
                             ' provided {}'.format(frac))
        arr = self.cmap(frac)
        if kind == 'rgb':
            if as_string is False:
                pass
            else:
                arr = [tuple(i) for i in arr[:, :-1] * 255]
                arr = cl.to_rgb(arr)
        elif kind == 'husl':
            if as_string is False:
                arr = np.array([pl.husl.rgb_to_husl(r, g, b)
                                for r, g, b, _ in arr])
            else:
                arr = [tuple(i) for i in arr[:, :-1] * 255]
                arr = cl.to_hsl(arr)
        elif kind == 'html':
            arr = [tuple(i) for i in arr[:, :-1] * 255]
            arr = cl.to_hsl(arr)
            arr = HTML(cl.to_html(arr))
        else:
            raise ValueError("Kind {} not supported".format(kind))
        return arr


def _add_middle_color(colors, midpoint, mid_spread=.3, log_amt=1e-3):
    """Converts the center of a colormap to a particular color."""
    mid_strings = dict(light=(.95, .95, .95),
                       dark=(.133, .133, .133))
    if isinstance(midpoint, str):
        if midpoint not in mid_strings.keys():
            raise ValueError('If midpoint is a string, must be'
                             ' one of %s' % mid_strings.keys())
        midpoint = mid_strings[midpoint]
    if len(midpoint) != 3:
        raise ValueError('Midpoint must be a tuple/list of length 3')
    if colors.ndim != 2:
        raise ValueError('Input colors must be 2d')
    if any([mid_spread <= 0, mid_spread >= 1]):
        raise ValueError('mid_spread must be between 0 and 1')

    # Find middle index
    ix_mid = int(colors.shape[0] / 2)
    n_mid_colors = int(colors.shape[0] * (mid_spread / 2.))

    # Get colors associated w/ the middle bounds
    num_col_hi = colors[ix_mid + n_mid_colors, :3]
    num_col_lo = colors[ix_mid - n_mid_colors, :3]

    # Iterate through colors and add the midpoitn to middle indices
    col_adds = []
    for col_side in [num_col_lo, num_col_hi]:
        # Iterate through rgb
        this_side = []
        for colmid, colend in zip(midpoint, col_side):
            if col_side is num_col_hi:
                lins = np.linspace(colmid, colend, n_mid_colors)
            else:
                lins = np.linspace(colend, colmid, n_mid_colors)
            this_side.append(lins)
        this_side = np.vstack(this_side)
        col_adds.append(this_side)
    col_adds = np.hstack(col_adds).T

    # Now overwrite the old colors
    weights = np.logspace(np.log10(log_amt), np.log10(1), n_mid_colors)
    weights = np.hstack([weights[::-1], weights])
    col_out = colors.copy()
    col_replace = col_out[ix_mid - n_mid_colors: ix_mid + n_mid_colors, :3]

    col_zip = zip(col_replace, col_adds, weights)
    colors_new = np.zeros([len(col_replace), 3])
    for i, (cold, cnew, cweight) in enumerate(col_zip):
        colors_new[i] = np.average([cold, cnew], axis=0,
                                   weights=[cweight, 1-cweight])

    col_out[ix_mid - n_mid_colors: ix_mid + n_mid_colors, :3] = colors_new
    return col_out


# Color auto-names
def _names_to_rgb(names):
    import webcolors as wb
    rgb = np.array([wb.name_to_rgb(i) for i in names])
    return rgb / 255.


def _closest_color_from_rgb(rgb):
    """Pulled from http://stackoverflow.com/questions/9694165/
    convert-rgb-color-to-english-color-name-like-green
    """
    import webcolors as wb
    min_colors = {}
    for key, name in wb.css3_hex_to_names.items():
        base_color = np.asarray(wb.hex_to_rgb(key))
        diff = base_color - np.asarray(rgb)
        min_colors[np.sum(diff ** 2)] = name
    return min_colors[np.min(list(min_colors.keys()))]


def _get_color_names(array):
    """Use webcolors to get the closest named color for rgb values"""
    import webcolors as wb
    array = np.atleast_2d(array)
    color_names = []
    for rgb in array:
        try:
            i_name = wb.rgb_to_name(rgb)
        except ValueError:
            i_name = _closest_color_from_rgb(rgb)
        color_names.append(i_name)
    return color_names
