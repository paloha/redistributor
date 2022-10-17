"""
.. include:: readme.md
"""

from __future__ import division

import warnings
import numpy as np
from scipy.stats import rv_continuous
from scipy.interpolate import interp1d
from sklearn.base import TransformerMixin
from sklearn.neighbors import KernelDensity as ScikitKDE
from scipy.stats import norm

class Redistributor(TransformerMixin):
    """
    An algorithm for automatic transformation of data from arbitrary
    distribution into arbitrary distribution. Source and target distributions
    can be known beforehandand or learned from the data using
    LearnedDistribution class. Transformation is piecewise linear, monotonic
    and invertible.

    Implemented as a Scikit-learn transformer. Can be fitted on 1D vector
    and saved to be used later for transforming other data assuming the same
    source distribution.

    Uses source's and target's `cdf()` and `ppf()` to infer the
    transform and inverse transform functions.

    `transform_function = target_ppf(source_cdf(x))`
    `inverse_transform = source_ppf(target_cdf(x))`
    """

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def fit(x=None, y=None):
        """
        Redistributor does not need to be fitted.
        """
        pass

    def transform(self, x):
        """
        Transform the data from source to target distribution.
        """
        return self.target.ppf(self.source.cdf(x))

    def inverse_transform(self, x):
        """
        Inverse transform the data from target to source distribution.
        """
        return self.source.ppf(self.target.cdf(x))

    def kstest(self, n=20):
        """
        Performs the (one-sample or two-sample) Kolmogorov-Smirnov test.
        """
        from scipy.stats import kstest
        return kstest(self.source.rvs, self.target.cdf, N=n,
                      alternative='two-sided', mode='auto')

    def plot_transform_function(self, bins=1000, newfig=True, figsize=(16, 5)):
        """
        Plotting the learned transformation from source to target.
        """
        import matplotlib.pyplot as plt
        x = np.linspace(*self.source._get_support(), bins)
        t = self.transform(x)
        if newfig:
            plt.figure(figsize=figsize)
        plt.title('Transform function')
        plt.plot(x, t)
        if newfig:
            plt.show()
            plt.close()
        return


class LearnedDistribution(rv_continuous):
    """
        A continuous random variable obtained by estimating the empirical
        distribution of a user provided 1D array of numeric data `x`. It
        can be used to sample new random points from the learned distribution.

        It approximates the Cumulative Distribution Function (`cdf`) and
        Percent Point Function (`ppf`) of the underlying distribution of `x`
        using linear interpolation on a lattice.

        An approximation of the Probability Density Function (`pdf`) is
        computed as an interpolation of the numerical derivative of the `cdf`
        function. Please note it can oscilate a lot if `bins` is high.

        The distribution is defined on a closed finite interval `[a, b]` or
        `[xmin, xmax]` or combination thereof, depending on which bound/s
        were specified by the user.

        WARNING: It can not be used to learn discrete distributions.


        Parameters
        ----------

        x : 1D numpy array
            Values from which the distribution will be estimated.
            The size of the array should be rather large, in case
            you have too small sample, consider using KDE class instead.
            Large magnitude of the array values in combination with
            small amount of samples, e.g.

        a : numeric or None
            Left boundary of the distribution support if known.
            If specified, must be smaller than `x.min()`.

        b : numeric or None
            Right boundary of the distribution support if known.
            If specified, must be bigger than `x.max()`.

        bins : int or None
            User specified value of bins. Min is 3, max is `x.size`.
            If None or 0, bins are set automatically. Upper bound
            is set to 5000 to prevent unnecessary computation.
            Used to specify the density of the lattice. More bins
            means higher precision but also more computation.

        keep_x_unchanged : bool, default True
            If True, the `x` array will be copied before partial sorting.
            This will result in increased memory usage. But it will
            not reorder the user provided array.

            If False, there will not be any additional memory consumption.
            But the user provided array `x` might change its order.
            This might be very useful if `x` is a large array and there is
            not enough available memory.

        subsample_x : int, default None
            Sacrifice precision for speed by first subsampling array `x`
            with a defined integer step. Not doing `random.choice()` but rather
            simple `slice(None, None, subsample_x)` because it is faster and
            we assume the array is randomly ordered. Can lead to significant
            speedups. If you need different approach to subsampling, do it in
            advance, provide already subsampled `x` and set this to None.

        ravel_x : bool, default True
            LearnedDistribution requires 1D arrays. So the `x` is by default
            flattened to 1D using `np.ravel()`.

        assume_sorted : bool, default False
            If the user knows that `x` is sorted, setting this to True will
            save computation by ommiting partial sorting the array.
            Especially useful if the array `x` is big. E.g. 1GB of data
            takes approx. 10s to partial sort on 5000 positions.
            If `False` and `x` is almost sorted, it will still be faster than
            if `x` is randomly ordered.

        fill_value : None, float, 2-tuple, 'auto', default='auto'
            Specifies where to map the values out of the `cdf` support. See the
            docstring of scipy.interpolate.interp1d to learn more about the
            possible options. Additionally, this class enables the user to use
            the default `auto` option, which sets reasonable `fill_value`
            automatically.

            WARNING: Not all choices of `fill_value` that are possible are also
            valid. E.g. `fill_value` should not be manually set to value
            smaller than 0 or larger than one. Also, `fill_value` should
            not be set such that it would make the output function decreasing.
            This also rules out the usage of 'extrapolate' option. All of these
            choices would not lead to a meaningful output in terms of
            a Cumulative Distribution Function.

        bounds_error : bool or 'warn', default 'warn'
            If True, raises an error when values out of `cdf` support are
            encountered. If False or 'warn', the invalid values are mapped to
            `fill_value`. For more details see the docstring of class
            `interp1d_with_warning`.

        resolve_duplicates : 2-tuple (`dist`, `mode`) or None,
                             default ('max', 'raise')
            If not None, makes a call to `make_unique` with specified `dist`
            and `mode` to make sure all `lattice_values` are unique. Read more
            in the docstring of `make_unique` function.

            WARNING: If None, the array is kept with duplicates which means the
            `p != cdf(ppf(p))`. In case there is mulitple duplicates of `xmin`
            or `xmax` values, `cdf(xmin)` will fail to map to Δ and `cdf(xmax)`
            will fail to map to 1 - Δ as it should.

        name : str, default 'LearnedDistribution'
            Name of the instance. Useful for locating source of warnings, etc.

        seed : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, default None
            See the docstring of scipy.stats.rv_continuous.
            Used in `make_unique()` and `rvs()`.

        kwargs : all other keyword arguments accepted by rv_continous.
        """

    def __init__(self, x, a=None, b=None, bins=None, keep_x_unchanged=True,
                 subsample_x=None, ravel_x=True, assume_sorted=False,
                 fill_value='auto', bounds_error='warn',
                 resolve_duplicates=('max', 'raise'),
                 seed=None, name='LearnedDistribution',
                 **kwargs):

        super().__init__(name=name, seed=seed, **kwargs)

        if ravel_x:
            x = x.ravel()

        # Sacrifice precision for speed
        if isinstance(subsample_x, int):
            if 2 <= subsample_x <= x.size:
                x = x[::subsample_x]
            else:
                raise ValueError('Not 2 <= subsample_x <= x.size.')

        # Handling input data and interval
        self._validate_x(x)
        self.xmin = x.min()
        self.xmax = x.max()
        self._validate_a_b(a, b)
        self.a = a
        self.b = b

        # Arguments for interpolation
        self.bounds_error = bounds_error
        self.fill_value = fill_value

        # Setting lattice density
        self.bins = self._infer_bins(x.size, bins)

        # Argument for treating duplicates
        self.resolve_duplicates = resolve_duplicates

        # Interpolating to get the empirical distribution
        self.assume_sorted = assume_sorted
        lattice, vals = self._get_lattice_and_vals(x, keep_x_unchanged)
        self._cdf = self._get_cdf(lattice, vals)
        self._ppf = self._get_ppf(lattice, vals)
        self._pdf = self._get_pdf()

    def _get_support(self, *args):
        """
        Support of LearnedDistribution `cdf` does not depend on any scipy
        argument, we keep args only to keep the signature unchanged from super.

        In this case, the support of `cdf` depends only on whether `a` and/or
        `b` were specified explicitely or as None values. Here, we return the
        "valid" support based on data. The "valid" support might be shrunk if
        either `a` or `b` is set to None. Only the points from the "valid"
        support actually map to unique values. So, if the "valid" support is
        not `[a, b]` but e.g. `[xmin, b]`, all the points from the interva
        `[a, xmin]` will map to the same value.

        In case the boundaries were not set, `self.a` and/or `self.b` are kept
        stored as Nones to keep the information about the object config for
        future reference.

        Returns
        -------
        a, b : numeric (float, or int)
            End-points of the valid `cdf` support.
        """
        return self._cdf.x[0], self._cdf.x[-1]

    def _get_support_ppf(self, *args):
        """
        The support of `ppf` in scipy is always `[0, 1]` so this method does
        not exist in `rv_continuous`. Here, we return the "valid" support based
        on data. The "valid" support might be shrunk if either `a` or `b` is
        set to None. However, all the values from `[0, 1]` are actually
        supported. Although, only the points from the "valid" support actually
        map to unique values. So, if the "valid" support is not `[0, 1]` but
        e.g. `[Δ, 1]`, all the points from the interval `[0, Δ]` will map to
        the same value.

        Returns
        -------
        a, b : numeric (float, or int)
            End-points of the valid `ppf` support.
        """
        return self._ppf.x[0], self._ppf.x[-1]

    def _validate_x(self, data):
        """
        Validation of the input data.

        Parameters
        --------
        data: numpy array of data to be validated
        """
        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError('Input array dtype must be floating point.')
        if np.issubdtype(data.dtype, np.float16):
            warnings.warn((
                'Using float16 data can lead to large errors. '
                'Rather use f32 or f64.'))
        if not data.ndim == 1:
            raise ValueError('Input array must be 1D. You can use x.ravel().')

    def _validate_a_b(self, a, b):
        """
        Validation of the boundaries.

        Parameters
        --------
        a: numeric or None
            See the docstring of __init__().
        b: numerc or None
            See the docstring of __init__().
        """
        if a is not None:
            if not np.isfinite(a):
                raise ValueError(f'a {a} must be finite.')
            if a >= self.xmin:
                raise ValueError(f'a {a} must be < than xmin {self.xmin}.')
        if b is not None:
            if not np.isfinite(b):
                raise ValueError(f'b {b} must be finite.')
            if b <= self.xmax:
                raise ValueError(f'b {b} must be > than xmax {self.xmax}.')

    def _infer_bins(self, n, bins):
        """
        Infers and validates the number of bins.

        Parameters
        --------
        n: int
            Size of the data.

        bins: int or None
            See the docstring of __init__().
        """
        if bins is None or bins == 0:
            bins = min(n, int(5e3))
        if bins > n or bins < 3:
            raise ValueError(f'Bins ({bins}) must be 2 < bins <= x.size')

        return bins

    def _get_lattice_and_vals(self, x, keep_x_unchanged):
        """
        Creating the `lattice` based on the number of `bins` and assembling the
        corresp. `lattice_vals` from provided array `x` using partial sort.

        Parameters
        --------
        x: 1D numpy array
            See the docstring of __init__().

        keep_x_unchanged: bool
            See the docstring of __init__().

        Returns
        -------

        lattice, 1D array of equidistant values
            Support of ppf or range of cdf. The first value of the array will
            be either `0` or `Δ` and the last value will be either `1` or
            `1 - Δ` depending if `a` or `b` are specified or None. Size of the
            array will range from `bins` to `bins + 2`. `Δ = 1 / (bins + 1)`.

        lattice_vals, 1D array
            Range of ppf or support of cdf. The first value of the array will
            be either `xmin` or a and the last value will be either `xmax` or b
            depending if a or b are specified or None. Size of the array will
            be the same as of `lattice`.
        """

        # Indices at which we need x to be sorted
        indices = np.linspace(
            0, x.size - 1, self.bins).round().astype(int)

        if not self.assume_sorted:
            # If bins is reasonably small in comparison to x.size it
            # pays off to do partial introselect sort instead of full sort
            use_partial = (self.bins / x.size) <= 0.25
            if keep_x_unchanged:  # Does not change x but uses twice the memory
                x = np.partition(x, indices) if use_partial else np.sort(x)
            else:  # Doesn't use additional memory but alters the order of data
                x.partition(indices) if use_partial else x.sort()

        # Expand the cdf suport by a and/or b from Left and/or Right side?
        L, R = self.a is not None, self.b is not None

        # Get the values to build the lattice
        vals_at_indices = x[indices]
        if self.resolve_duplicates is not None:
            make_unique(vals_at_indices, *self.resolve_duplicates,
                        assume_sorted=True, inplace=True,
                        random_state=self.random_state)

        # Get the _ppf.y (or _cdf.x) values
        lattice_vals = np.hstack([[self.a] * L, vals_at_indices, [self.b] * R])

        # Get the _ppf.x (or _cdf.y) values
        delta = 1 / (self.bins + 1)  # Shrink using Δ to exclude 0 or 1
        lattice = np.linspace(
            delta * (not L), 1 - delta * (not R), lattice_vals.size)

        if not all(np.isfinite(lattice_vals)):
            raise ValueError('Values of x on the lattice must be finite.')

        return lattice, lattice_vals

    def _get_cdf(self, lattice, lattice_vals):
        """
        Interpolates the lattice on lattice_vals to get the `cdf`.
        If the function is evaluated at a point outside of the support,
        it is mapped to fill_value.

        Table of `fill_value` if `self.fill_value == 'auto'`
        Δ = 1 - (bins + 1)
        ------------------------------------------------------------------
        a     | b     | cdf support | truncated to | fill_value
        ------------------------------------------------------------------
        None  | None  | xmin, xmax  | xmin, xmax   | Δ, 1-Δ
        None  | b     | xmin, b     | xmin, None   | Δ, None
        a     | None  | a,    xmax  | None, xmax   | None, 1-Δ
        a     | b     | a,    b     | None, None   | None, None
        """

        if self.fill_value == 'auto':
            fill_value = (lattice[0] if self.a is None else None,
                          lattice[-1] if self.b is None else None)
        else:
            # User defined fill_value
            fill_value = self.fill_value

        return interp1d_with_warning(
            lattice_vals, lattice, kind='linear', assume_sorted=True,
            bounds_error=self.bounds_error, fill_value=fill_value,
            name=f'{self.name} cdf interpolant')

    def _get_ppf(self, lattice, lattice_vals):
        """
        Interpolates the lattice_vals on a lattice to get the `ppf`.
        `fill_values` is set to the `xmin` and `xmax` to avoid problems
        when generating random sample with `rvs()`. `bounds_error` is
        set to `False` because user does not need a warning about this
        behaviour and values `p < 0` or `p > 1` should not ever occur at all.
        """
        fill_value = (lattice_vals[0], lattice_vals[-1])
        return interp1d_with_warning(
            lattice, lattice_vals, kind='linear', assume_sorted=True,
            bounds_error=False, fill_value=fill_value,
            name=f'{self.name} ppf interpolant')

    def cdf(self, q):
        """
        Interpolates the lattice on lattice_vals to get the
        piecewise linear approximation to the emprical cumulative
        distribution function of the learned distribution.

        Parameters
        ----------
        q : array_like
            quantile

        Returns
        -------
        p : 1D numpy array of floats
            Cumulative distribution function evaluated at `q`.
            I.e. lower tail probability corresponding to the quantile q.
        """
        # We do not need the default argument checking from scipy
        # because we handle the invalid values differently using
        # the class `interp1d_withwarning`. Therefore:
        q = np.asarray(q)
        return self._cdf(q)

    def ppf(self, p):
        """
        Interpolates the lattice_vals on lattice to get the
        piecewise linear approximation to the inverse of the emprical
        cumulative distribution function of the learned distribution.
        I.e. a Percent point function of the learned distribution.

        Parameters
        ----------
        p : array_like
            lower tail probability

        Returns
        -------
        q : 1D numpy array of floats
            Percent point function evaluated at `p`.
            I.e. quantile corresponding to the lower tail probability p.
        """
        # We do not need the default argument checking from scipy
        # because we handle the invalid values differently using
        # the class `interp1d_withwarning`. Therefore:
        p = np.asarray(p)
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError('Some values out of ppf support [0, 1].')
        return self._ppf(p)

    def _get_pdf(self):
        """
        Interpolates a derivative of cdf to obtain the pdf.
        """
        g = np.linspace(*self._get_support(), self.bins)
        c = self.cdf(g)  # Evaluated cdf
        dx = 1 / (g[1] - g[0])
        dif = np.round(np.ediff1d(c, to_begin=c[1] - c[0]) * dx, decimals=10)
        return interp1d_with_warning(g, dif, kind='linear', assume_sorted=True,
                                     name=f'{self.name} pdf interpolant')

    def _entropy(self, *args):
        """
        Differential entropy of the learned RV.
        """
        from scipy.special import entr
        from scipy.integrate import simpson
        g = np.linspace(*self._get_support(), self.bins)
        return simpson(entr(self._pdf(g)), x=g)

    def rvs(self, size, random_state=None):
        """
        Random sample from the learned distribution.
        """
        if random_state is None or type(random_state) is int:
            from numpy.random import default_rng
            random_state = default_rng(random_state)
        return self.ppf(
            random_state.uniform(*self._get_support_ppf(), size=size))


def plot_cdf_ppf_pdf(dist, a=None, b=None, bins=None,
                     v=None, w=None, rows=1, cols=3,
                     figsize=(16, 5)):
    """
    Just a convinience function for visualizing the dist
    `cdf`, `ppf` and `pdf` functions.

    Parameters
    ----------
    a: float
        Start of the cdf support
    b: float
        End of the cdf support
    v: float
        Start of the ppf support
    w: float
        End of the ppf support
    rows: int,
        Number of rows in the figure
    cols: int,
        Number of cols in the figure
    figsize: None or tuple
        If None, no new figure is created.
    """
    import matplotlib.pyplot as plt
    if a is None:
        a = dist._get_support()[0]
    if b is None:
        b = dist._get_support()[1]
    if bins is None:
        if hasattr(dist, 'bins'):
            bins = dist.bins
        elif hasattr(dist, 'grid_density'):
            bins = dist.grid_density
        else:
            bins = 1000

    if v is None:
        v = dist._get_support_ppf()[0]
    if w is None:
        w = dist._get_support_ppf()[1]

    x = np.linspace(a, b, bins)
    y = np.linspace(v, w, bins)

    if figsize is not None:
        plt.figure(figsize=figsize)
        plt.tight_layout()

    plt.subplot(rows, cols, 1)
    plt.title(f'{dist.name} CDF')
    plt.plot(x, dist.cdf(x))

    plt.subplot(rows, cols, 2)
    plt.title(f'{dist.name} PPF')
    plt.plot(y, dist.ppf(y))

    plt.subplot(rows, cols, 3)
    plt.title(f'{dist.name} PDF')
    plt.plot(x, dist.pdf(x))
    plt.ylim(-0.001, None)

    if figsize is not None:
        plt.show()
        plt.close()
        return
    else:
        return plt


class interp1d_with_warning(interp1d):
    """
    By default behaves exactly as scipy.interpolate.interp1d but allows
    the user to specify `bounds_error = 'warn'` which overrides the
    behaviour of `_check_bunds` to warn instead of raising an error.

    Parameters
    ----------
    Accepts all the args and kwargs as scipy.interpolate.interp1d.
    """

    def __init__(self, *args, **kwargs):
        self.warn = False
        self.name = kwargs.pop('name', 'Interp1D')
        bounds_error = kwargs.get('bounds_error')
        if bounds_error == 'warn':
            self.warn = True
            bounds_error = True
        super().__init__(*args, **kwargs)

    def _check_bounds(self, x_new):
        """
        Overriding the _check_bounds method of scipy.interpolate.interp1d
        in order to provide a functionality of warning the user instead of
        just throwing an error when some value is out of bounds. Even if
        fill_value is specified a warning can be issued to let the user
        know it was necessary to use the fill_value and from which side.
        """

        below_bounds = x_new < self.x[0]  # Find values which are bellow bounds
        above_bounds = x_new > self.x[-1]  # Find values which are above bounds

        msg = ("{}: {} out of {} values in x_new are {} the interpolation "
               "range. Read the docs of `fill_value` and `bounds_error` "
               "to manage the behavior.")

        if below_bounds.any():
            if self.bounds_error:
                m = msg.format(self.name, below_bounds.sum(),
                               below_bounds.size, 'below')
                if self.warn:
                    m += (' Mapping the invalid values to value: '
                          f'{self._fill_value_below}.')
                    warnings.warn(m)
                else:
                    raise ValueError(m)

        if above_bounds.any():
            if self.bounds_error:
                m = msg.format(self.name, above_bounds.sum(),
                               above_bounds.size, 'above')
                if self.warn:
                    m += (' Mapping the invalid values to value: '
                          f'{self._fill_value_above}.')
                    warnings.warn(m)
                else:
                    raise ValueError(m)

        return below_bounds, above_bounds


class KernelDensity():
    """
    Wrapper around KernelDensity for ease of use as a source or
    target distribution of Redistributor. It extends the KDE by
    providing cdf and ppf functions.

    Only supports 1D input because Redistributor also works only in 1D.
    Only supports gaussian kernel. CDF supports two methods, precise and fast.
    CDF precise is computed using a formula. CDF fast is a linear interpolation
    of the CDF precise on a grid of specified density. There is no explicit
    formula for PPF of gaussian mixutre, so here it is approximated using
    linear interpolation of the CDF precise on a grid of specified density.

    Parameters
    ----------

    x : numeric or 1D numpy array
        1D vector of which the distribution will be estimated.

    ravel_x : bool, default True
        KDE requires 1D arrays. So the `x` is by default
        flattened to 1D using `np.ravel()`.

    grid_density : int
        User specified number of grid points on which the CDF is computed
        precisely in order to build the interpolants for fast CDF and PPF.
        The same grid is used for CDF fast and PPF. The user specified
        value of grid_density is not it's final value. It is updated
        during initialization of this object on call of `self._get_ppf()`.

    cdf_method : str, one of {'precise', 'fast'}
        Specifies the default method to be used when self.cdf() is called.
        'precise' computes cdf using a formula, 'fast' uses a precomputed
        interpolant to get a fast approximation.

    name : str, default 'LearnedDistribution'
            The name of the instance.

    kwargs : keyword arguments accepted by sklearn.neighbors.KernelDensity.


    Methods
    -------

    pdf : Probability Density Function of a Gaussian Mixture
    cdf : Cumulative Distribution Function of a Gaussian Mixture
    ppf : Approximation of a Percent Point Function of a Gaussian Mixture
    rvs : Random sample generator
    """

    def __init__(self, x, ravel_x=True, grid_density=int(1e4),
                 cdf_method='fast', name='KDE', **kwargs):

        self.name = name

        if kwargs.get('kernel') not in [None, 'gaussian']:
            raise ValueError('Only gaussian kernel is supported here.')

        # Fitting the KDE
        x = np.asarray(x)
        if ravel_x:
            x = x.ravel()
        self._validate_shape(x)
        self.kde = ScikitKDE(**kwargs).fit(x.reshape(-1, 1))

        # Approximation of cdf with linear interpolation
        self.cdf_method = cdf_method
        self._cdf_fast = None  # Computed during call of _get_ppf()

        # Controls the ppf approximation precision
        if grid_density < 10:  # 10 is already unreasonably small
            raise ValueError('Grid density too small.')
        self.grid_density = grid_density
        self._ppf = self._get_ppf()

    def _validate_shape(self, data):
        if not data.ndim == 1:
            raise ValueError('Input array must be 1D. You can use x.ravel().')

    def _get_support(self, *args):
        return self.a, self.b

    def _get_support_ppf(self, *args):
        return self.ppfa, self.ppfb

    def rvs(self, size=1, random_state=None):
        """
        Random sample from the estimated distribution.
        """
        return self.kde.sample(size, random_state).ravel()

    def pdf(self, x):
        """
        Probability density function of the estimated distribution.

        Parameters
        ----------
        q : array_like
            quantile

        Returns
        -------
        d : 1D numpy array of floats
            Probability density function evaluated at `q`.
            I.e. probability density corresponding to the quantile q.
        """

        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1)
        self._validate_shape(x)
        return np.exp(self.kde.score_samples(x.reshape(-1, 1)))

    def cdf(self, q, method=None):
        """
        Cumulative distribution function of the estimated distribution.

        Parameters
        ----------
        q : array_like
            quantile

        Returns
        -------
        p : 1D numpy array of floats
            Cumulative distribution function evaluated at `q`.
            I.e. lower tail probability corresponding to the quantile q.
        """
        method = method or self.cdf_method

        if method == 'fast':
            fast = self._cdf_fast(q)
            # Handling out of support q by computing it precisely
            # out_of_support.sum() gives the number of out_of_support values
            out_of_support = np.isnan(fast)
            if out_of_support.any():
                fast[out_of_support] = \
                    self.cdf(q[out_of_support], method='precise')
            return fast

        elif method == 'precise':
            data = np.asarray(self.kde.tree_.data)
            out = 0.0
            for data_point in data:
                out += norm.cdf(q, loc=data_point, scale=self.kde.bandwidth)
            return out / data.size

        else:
            raise NotImplementedError('Method not it {"fast", "precise"}')

    def ppf(self, p):
        """
        Percent point function of the estimated distribution.
        This method approximates the ppf based on linear interpolation
        of the cdf on self.grid_density many points. There is no formula
        for precise computation of gaussian mixture ppf. Therefore, if
        we wanted a precise function, we would need to bisect the cdf.
        Bisecting is very slow in comparison to just computing the cdf
        on a grid and using the interpolant to approximate the ppf.

        Parameters
        ----------
        p : array_like
            lower tail probability

        Returns
        -------
        q : 1D numpy array of floats
            Percent point function evaluated at `p`.
            I.e. quantile corresponding to the lower tail probability p.
        """
        p = np.asarray(p)
        if (p < 0).any() or (p > 1).any():
            raise ValueError('Value in p out of PPF support (0, 1).')
        return self._ppf(p)

    def _get_ppf(self):
        """
        In order to get a fast ppf function, we will sample the cdf on its
        support using a grid of desired density. Then we create an interpolant
        which maps the values inversly, thus getting an approximation to a ppf.

        Theoretically, ppf(0), ppf(1) must map to -inf, +inf. We can not
        interpolate to infinite values, therefore, for the ppf range, we pick
        the closest finite values from the grid. Everything outside will be
        set to ±inf. This way we obtain a precise enough interpolant which
        also maps the 0 and 1 to ±inf.
        """

        def argvalid(arr):
            """
            Returns 2-tuple of indices `first` and `last` which can be used to
            slice `arr` to get only valid values (0 < valid < 1). Function
            assumes `arr` is sorted and finite. If all values from left side
            are valid, first = 0. If all values from right side are valid,
            last = None.
            """
            v = arr > 0
            w = arr < 1
            first = v.size - v.sum()
            last = w.sum() - w.size
            last = None if last == 0 else last
            return first, last

        # Computing the empirical range of the ppf
        # Bandwidth * 39 is just an empirical distance from the mean
        # of a gaussian which maps to 0 due to floating point precision.
        # Therefore a, b is just a bit bigger than an empirical
        # ppf range (or cdf support).

        # Approximate a, b
        a = np.min(self.kde.tree_.data) - self.kde.bandwidth * 39
        b = np.max(self.kde.tree_.data) + self.kde.bandwidth * 39

        # Since a, b is a bit bigger than the cdf support
        # a few of the first cdfx values will all map to 0
        # and a few of the last cdfx values will all map to 1.
        # We have to remove those duplicate gridpoints
        # to properly define our interpolant (from cdf to ppf).
        cdfx = np.linspace(a, b, self.grid_density)  # grid
        cdfy = self.cdf(cdfx, method='precise')
        first, last = argvalid(cdfy)

        if np.nansum(np.abs(np.array([first, last], dtype=float))) == cdfy.size:
            raise ValueError('All grid points between a, b are invalid.')

        # Final x, y
        cdfx = cdfx[first:last]
        cdfy = cdfy[first:last]

        # New valid a, b (support of the cdf)
        self.a, self.b = cdfx[0], cdfx[-1]

        # Support of the ppf
        self.ppfa, self.ppfb = cdfy[0], cdfy[-1]

        # Real grid density (some points might have been excluded as invalid)
        self.grid_density = cdfx.size

        # Since the cdf is already evaluated, we can also just store it
        # to use it for fast cdf approximation (self.cdf(method='fast')
        self._cdf_fast = interp1d(cdfx, cdfy, bounds_error=False,
                                  fill_value=(np.nan))

        return interp1d(cdfy, cdfx, bounds_error=False,
                        fill_value=(-np.inf, np.inf))


def make_unique(array, dist='max', mode='raise', assume_sorted=True,
                inplace=False, random_state=None):
    """
    UTILITY FUNCTION TO FORCE LATTICE VALUES TO HAVE NON-REPEATING ELEMENTS

    Finds duplicate values in `array` and shifts them at most by `dist`
    to get an array of all unique values. Shifts are sampled randomly from
    uniform distribution.

    If `dist` is not smaller or equal to half the smallest distance between
    two non-duplicates, a duplicate point + noise could "jump behind" the next
    non-duplicate. E.g. for array [0, 1, 1, 2, 3] and `dist` = 1.5 the result
    could be np.sort([0, 1, 2.5, 2, 3]), i.e. the second occurrence of number 1
    was augmented by noise of 1.5 magnitude and in result it jumped to
    position 2.5 which is larger than 2, which was one of the original
    non-duplicate values. (This is an extreme example)

    NOTICE: there is no good way to implement this function as it changes the
    provided data to fullfill the assumption on non-repeating values. Whether
    it is a good idea to do it this way or some other way highly depends on
    use case. So make sure you know what you are doing.

    Parameters
    ----------

    array : 1D numpy array
        Array with potential of having duplicate elements.

    dist : float or 'max', default 'max'
        Max allowed shift of a duplicate point. If 'max' is used
        the `max_dist` = 1/2 min distance between two non-duplicates.

    mode : one of {'raise', 'clip', 'ignore', 'warn'}, default 'raise'
        Behavior when specified `dist` is larger than `max_dist`.
        'raise'  - raises a ValueError
        'clip'   - clips the `dist` to `max_dist`
        'ignore' - will use `dist` no matter the consequences, use with caution
        'warn'   - same as ignore, just a warning is issued

    assume_sorted : bool, default True
        If not, we sort at the beginning.

    inplace : bool, default False
        If True, adjust array inplace, otherwise make a copy.

    random_state : RandomState, int, or None, default None
        Seed or generator for noise generation.

    Returns
    -------

    array: sorted 1D numpy array with no duplicates
        If `inplace=True`, returns None
    """

    if not inplace:
        array = array.copy()

    if not assume_sorted:
        array.sort()

    _min, _max = array[0], array[-1]
    diff = np.ediff1d(array, to_end=np.abs(_max))

    if assume_sorted and np.any(diff < 0):
        raise ValueError('If `assume_sorted=True` array must be sorted.')

    # Find all duplicates (bool mask with True if element is a duplicate)
    dupl = diff == 0

    # No work if no duplicates
    n_duplicates = dupl.sum()
    if n_duplicates == 0:
        return array

    # Min and Max must be treated separately
    where_min = array == _min
    where_max = array == _max

    # Choosing the dist
    max_dist = diff[np.logical_not(dupl)].min() / 2
    dist = max_dist if dist == 'max' else dist

    # Choosing the magnitude of the uniform noise
    if dist > max_dist:
        if mode == 'raise':
            raise ValueError(
                f'dist > max_dist ({dist} > {max_dist}). '
                'Make dist smaller or manage this behavior '
                'by changing the value of the mode argument.')
        elif mode == 'clip':
            dist = max_dist
        elif mode == 'warn':
            warnings.warn(f'dist > max_dist ({dist} > {max_dist}).')
        elif mode == 'ignore':
            pass
        else:
            raise NotImplementedError(f'Passing mode={mode} is not supported.')

    if isinstance(random_state, int) or random_state is None:
        random_state = np.random.RandomState(seed=random_state)

    # Adding noise
    change = dupl & np.logical_not(where_min | where_max)
    array[change] += random_state.uniform(-max_dist, max_dist, change.sum())
    change = dupl & where_max
    array[change] += random_state.uniform(-max_dist, 0, change.sum())
    change = dupl & where_min
    array[change] += random_state.uniform(0, max_dist, change.sum())

    array.sort()
    if inplace:
        return
    return array


def save_redistributor(d, path):
    """Saves the Redistributor object to a file."""
    import joblib
    joblib.dump(d, path)


def load_redistributor(path):
    """Loads the Redistributor object from a file."""
    import joblib
    return joblib.load(path)
