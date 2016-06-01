"""Quickly translate between different color representations."""
import seaborn.palettes as pl
from seaborn import palplot
from IPython.display import HTML
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import colorlover as cl
import webcolors as wb

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
                    colors = np.array([wb.name_to_rgb(i) for i in colors])
                    colors = colors / 255.
            colors = pl.blend_palette(colors, as_cmap=True)
        self.cmap = colors

    def to_numeric(self, n_bins=100, kind='rgb'):
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

    def to_strings(self, n_bins=100, kind='rgb'):
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
            colors_string = [pl.husl.rgb_to_hex(i) for i in array]
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

    def to_diverging(self, as_cmap=True, center='light', **kwargs):
        """Convert diverging colormap from first/last color in self.

        Parameters
        ----------
        as_cmap : bool
            Whether to return the diverging map as a list of colors
            or a LinearSegmentedColormap
        center : 'light' | 'dark'
            Whether the center color is dark or light.

        Returns
        -------
        colors_div : list of arrays | matplotlib colormap
            The diverging colormap, created by interpolating between
            the first and last colors in self.cmap.
        """
        colors_rgb = self.cmap([0., 1.])
        colors_husl = [pl.husl.rgb_to_husl(r, g, b)[0]
                       for r, g, b, a in colors_rgb]
        colors_div = pl.diverging_palette(colors_husl[0], colors_husl[1],
                                          center=center, **kwargs)
        colors_div = pl.blend_palette(colors_div, as_cmap=as_cmap)
        return colors_div

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


def _closest_color_from_rgb(rgb):
    """Pulled from http://stackoverflow.com/questions/9694165/
    convert-rgb-color-to-english-color-name-like-green
    """
    min_colors = {}
    for key, name in wb.css3_hex_to_names.items():
        base_color = np.asarray(wb.hex_to_rgb(key))
        diff = base_color - np.asarray(rgb)
        min_colors[np.sum(diff ** 2)] = name
    return min_colors[np.min(min_colors.keys())]


def _get_color_names(array):
    array = np.atleast_2d(array)
    color_names = []
    for rgb in array:
        try:
            i_name = wb.rgb_to_name(rgb)
        except ValueError:
            i_name = _closest_color_from_rgb(rgb)
        color_names.append(i_name)
    return color_names
