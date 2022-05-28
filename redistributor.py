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
            1D vector of which the distribution will be estimated.

        a : numeric or None
            Left boundary of the distribution support if known.
            If specified, must be smaller than x.min().

        b : numeric or None
            Right boundary of the distribution support if known.
            If specified, must be bigger than x.max().

        bins : int or None
            User specified value of bins. Min is 3, max is `x.size`.
            If None or 0, bins are set automatically. Upper bound
            is set to 1000 to prevent unnecessary computation.
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
            speedups.

        ravel_x : bool, default True
            LearnedDistribution requires 1D arrays. So the `x` is by default
            flattened to 1D using `np.ravel()`.

        assume_sorted : bool, default False
            If the user knows that `x` is sorted, setting this to True will
            save a most of time by ommiting partial sorting the array.
            Especially useful if the array `x` is big. E.g. 1GB of data
            takes approx. 10s to partial sort on 5000 positions.
            If `False` and `x` is almost sorted, it will still be faster than
            if `x` is randomly ordered.

        fill_value : None, array-like, float, 2-tuple or 'auto', default='auto'
            Specifies where to map the values out of the `cdf` support. See the
            docstring of scipy.interpolate.interp1d to learn more about the
            valid options. Additionally, this class enables the user to use
            the default `auto` option, which sets reasonable `fill_value`
            automatically.

        bounds_error : bool or 'warn', default 'warn'
            See the docstring of class interp1d_with_warning.

        dupl_method : str, one of {'keep', 'spread', 'cluster', 'noise'}
                      default 'spread'
            Method of solving duplicate lattice values. Read more in
            docstring of `make_unique()`.

        name : str, default 'LearnedDistribution'
            The name of the instance.

        seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, default None
            See the docstring of scipy.stats.rv_continuous.
            Used in `_prevent_same()` and `rvs()`.

        kwargs : all other keyword arguments accepted by rv_continous.


        Methods - TODO finish this documentation
        -------

        cdf
        ppf
        pdf
        rvs
        entropy
        ... fill in the rest which is implemented
        ... handle the rest which does not make sense
        """

    def __init__(self, x, a=None, b=None, bins=None, keep_x_unchanged=True,
                 subsample_x=None, ravel_x=True, assume_sorted=False,
                 fill_value='auto', bounds_error='warn', dupl_method='spread',
                 seed=None, name='LearnedDistribution', **kwargs):

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

        # Interpolating to get the empirical distribution
        self.assume_sorted = assume_sorted
        self.dupl_method = dupl_method
        lattice, vals = self._get_lattice_and_vals(x, keep_x_unchanged)
        self._cdf = self._get_cdf(lattice, vals)
        self._ppf = self._get_ppf(lattice, vals)
        self._pdf = self._get_pdf()

    def _get_support(self, *args):
        """
        Support of LearnedDistribution does not depend on any scipy arguments,
        we keep args only to keep the signature unchanged from super.

        The support depends only on whether `a` and/or `b` were specified
        explicitely or as Nones.

        `self.a` and/or `self.b` are kept stored as Nones to keep the
        information about the object config for future reference of the user.

        Returns
        -------
        a, b : numeric (float, or int)
            end-points of the distribution's support.
        """
        return self._cdf.x[0], self._cdf.x[-1]

    def _get_support_ppf(self, *args):
        """
        The support of `ppf` in scipy is always `[0,1]` so this method does not
        exist in `rv_continuous`. In our case, the support might be shrinked if
        any of the a, b is set to None.
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
            be either 0 or epsilon and the last value will be either 1 or
            1 - epsilon depending if a or b are specified or None. Size or the
            array will range from bins to bins + 2.

        lattice_vals, 1D array
            Range of ppf or support of cdf. The first value of the array will
            be either xmin or a and the last value will be either xmax or b
            depending if a or b are specified or None. Size of the array will
            be the same as of `lattice`.
        """

        # Do we need expansion by a or b from Left or Right side or both?
        L, R = self.a is not None, self.b is not None

        # Indices at which we need x to be sorted considering L and R
        indices = np.linspace(
            0, x.size + L + R - 1, self.bins).round().astype(int)
        indices = indices[L:-1 if R else None] - L

        if not self.assume_sorted:
            # Reorder values of x on indices as if the array was sorted
            if keep_x_unchanged:
                # Does not change x but uses twice the memory
                x = np.partition(x, indices)
            else:
                # Does not use additional memory but alters the order of data
                x.partition(indices)

        # Get the _ppf.y (or _cdf.x) values
        eps = 1 / (x.size + 1)  # Shrink the lattice with eps to exclude 0 or 1
        lattice_vals = np.hstack([[self.a] * L, x[indices], [self.b] * R])

        # Get the _ppf.x (or _cdf.y) values
        lattice = np.linspace(
            eps * (not L), 1 - eps * (not R), lattice_vals.size)

        if not all(np.isfinite(lattice_vals)):
            raise ValueError('Values of x on the lattice must be finite.')

        # If necessary, make duplicate values unique and warn the user.
        lattice_vals = make_unique(
            lattice_vals, self.random_state, mode=self.dupl_method)
        return lattice, lattice_vals

    def _get_cdf(self, lattice, lattice_vals):
        """
        Interpolates the lattice on lattice_vals to get the `cdf`.

        Table of `fill_value` if `self.fill_value == 'auto'` (n = x.size):
        ------------------------------------------------------------------
        a     | b     | cdf support | truncated to | fill_value
        ------------------------------------------------------------------
        None  | None  | xmin, xmax  | xmin, xmax   | 1/(n-1), (n-2)/(n-1)
        None  | b     | xmin, b     | xmin, None   | 1/(n-1), None
        a     | None  | a,    xmax  | None, xmax   | None, (n-2)/(n-1)
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
        behaviour and values `q < 0` or `q > 1` should not ever occur at all.
        """
        fill_value = (lattice_vals[0], lattice_vals[-1])
        return interp1d_with_warning(
            lattice, lattice_vals, kind='linear', assume_sorted=True,
            bounds_error=False, fill_value=fill_value, 
            name=f'{self.name} ppf interpolant')

    def cdf(self, k):
        # We do not need the default argument checking from scipy
        # because we handle the invalid values differently using
        # the class `interp1d_withwarning`. Therefore:
        k = np.asarray(k)
        return self._cdf(k)

    def ppf(self, q):
        # We do not need the default argument checking from scipy
        # because we handle the invalid values differently using
        # the class `interp1d_withwarning`. Therefore:
        q = np.asarray(q)
        if np.any(q < 0) or np.any(q > 1):
            raise ValueError('Some values out of ppf support [0, 1].')
        return self._ppf(q)

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


def save_redistributor(d, path):
    """Saves the Redistributor or Redistributor_multi object to a file."""
    import joblib
    joblib.dump(d, path)


def load_redistributor(path):
    """Loads the Redistributor or Redistributor_multi object from a file."""
    import joblib
    return joblib.load(path)


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
    Additionally, 
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
                m = msg.format(self.name, below_bounds.sum(), below_bounds.size, 'below')
                if self.warn:
                    m += (' Mapping the invalid values to value: '
                          f'{self._fill_value_below}.')
                    warnings.warn(m)
                else:
                    raise ValueError(m)

        if above_bounds.any():
            if self.bounds_error:
                m = msg.format(self.name, above_bounds.sum(), above_bounds.size, 'above')
                if self.warn:
                    m += (' Mapping the invalid values to value: '
                          f'{self._fill_value_above}.')
                    warnings.warn(m)
                else:
                    raise ValueError(m)

        return below_bounds, above_bounds


def make_unique(array, random_state, mode='spread', duplicates=None):
    """
    Takes a sorted array and adjusts the duplicate values such that all
    elements of the array are unique. The adjustment is done by linearly
    separating the duplicates. Read more in docsting of `_get_intervals`.

    In case `mode='keep'` this function does nothing and returns the array.

    Supports two deterministic modes 'spread' and 'cluster'. These two
    define onto how large interval the valueas are spread. If 'cluster'
    is not possible 'spread' is used implicitly.

    In case there are too many duplicates (>5e3), first uses addition of
    random noise to non-min and non-max values and then continues with the
    deterministic method.

    Keeps the min, max, and unique values unchanged.
    If the first iteration did not make all elements unique, repeats until
    failure and warns the user (should be rare).


    Parameters
    ----------
    array: 1D numpy array
        Sorted array with potential of having non-unique elements.

    random_state: RandomState

    mode: str, one of {'keep', 'spread', 'cluster', 'noise'}
      'keep' produces discontinuous cdf (cdf with vertical jumps)
          because it just simply keeps the non unique values
      'spread' is deterministic but slow to compute,
          it separates the non unique values equidistantly and
          tries to use all the available space between consecutive values.
      'cluster' is deterministic and also slow to compute,
          it separates the non unique values equidistantly but
          it does only use a small space around the value.
      'noise' is fast, very similar to 'cluster', but nondeterministic
          because it involves randomness and it handles min and max values
          separately to avoid jumping out of the a,b interval.

    duplicates: int, number of duplicates in previous iteration.
        Do not use, used only for recursion.


    Returns
    --------
    Sorted array of unique elements on the orignal interval.
    """

    if mode == 'keep':
        return array

    def _get_intervals(array, diff):
        """
        Iterates over the diff of the array, when it finds a duplicate
        value (i.e., when diff == 0), it adds it to the result dict and
        finds the interval onto which the duplicates can be spread such
        that the value does not jump over previous/next value or it's
        intrval. If two duplicate values are right after each other, they
        share the interval between them based on number of duplicates each
        of tham has. Note that with array [2, 2] the number of duplicates
        of the value 2 is counted as 1. The other is original.

        Returns
        -------
        dict {value: [n: int, n of val duplicates,
                      i: int, first index of duplicate value,
                      j: int, last index of duplicate value,
                      a: int, start of safe interval,
                      b: int, end of safe interval]}

        Example
        -------

        array = [0,0,1,3,4,6,6,6,7,7,9,9]
        diff  = [0,1,2,1,2,0,0,1,0,2,0,9]
        r = {
            0: [1, 0, 1, 0, 1],
            6: [2, 5, 7, 4, 6.666666666666667],
            7: [1, 8, 9, 6.666666666666667, 7.923076923076923],
            9: [1, 10, 11, 7.923076923076923, 9]}
        """
        r = {}
        n = 0
        prev = v = i = j = a = b = None
        for p, (d, dd) in enumerate(zip(diff, diff[1:])):
            if d == 0:  # Duplicate value
                v = array[p]
                if n == 0:  # First occurence
                    a = array[p] if p == 0 else array[p - 1]
                    i = p  # First occurence index
                n += 1
                if dd != 0:  # Last occurence
                    b = array[p + 2] if p + 2 < array.size else array[p + 1]
                    j = p + 1  # Last occurence index

                    # Two duplicates next to each other must share interval
                    # Proportion for each is assigned based on their counts
                    if prev is not None and v == r[prev][4]:
                        pn = r[prev][0]  # Previous n
                        d1 = prev - r[prev][3]  # Interval left of prev
                        d2 = v - prev  # Interval left of current value
                        d3 = b - v  # Interval right of current value
                        pw = (pn * d2) / (d1 + d2)  # weight of prev val
                        cw = (n * d2) / (d2 + d3)  # weight of current val
                        a = prev + (d2 * pw) / (pw + cw)
                        r[prev][4] = a  # Adjust previous val's b value

                    # Store result and restart counters
                    r[v] = [n, i, j, a, b]  # Mutable for adjustments
                    prev = v
                    v = i = j = a = b = None
                    n = 0
        return r

    eps = 1e3 * np.finfo(array.dtype).eps

    # Assuming sorted array
    _min, _max = array[0], array[-1]
    diff = np.ediff1d(array, to_end=np.abs(_max))
    if np.any(diff < 0):
        raise ValueError('Array must be sorted.')

    # Find all duplicates
    dupl = diff == 0

    # No work if no duplicates
    n_duplicates = dupl.sum()
    if n_duplicates == 0:
        return array

    if n_duplicates == duplicates:
        warnings.warn((
            f'Returning non-unique. Unable to remove {n_duplicates} '
            f'({(n_duplicates / array.size) * 100}%) duplicates.'))
        return array

    warnings.warn(
        (f'Adjusting {n_duplicates / array.size * 100}% non-unique '
         'lattice values. Avoid learning discrete distributions.'))

    if (n_duplicates > int(5e3) or mode == 'noise') and duplicates is None:
        warnings.warn((
            f'Array has too many duplicates ({n_duplicates}) '
            'to use the deterministic algorithm. Solving some '
            'or all by adding small random noise.'))
        change = dupl & np.logical_not((array == _min) | (array == _max))
        array[change] += random_state.uniform(eps, 10 * eps, change.sum())
        return make_unique(
            np.sort(array), random_state, mode, n_duplicates)
    else:
        # If there is not that many duplicates, we use
        # this method, which would be otherwise slower.
        intervals = _get_intervals(array, diff)
        for k, v in intervals.items():
            n, i, j, a, b = v
            if mode == 'cluster':
                a = np.max([a, k - n * eps])
                b = np.min([b, k + n * eps])

            # Using size n+1+n%2 to avoid a and k in lin
            lin = np.linspace(a, b, n + 1 + n % 2, endpoint=False,
                              dtype=array.dtype)[1 + n % 2:]
            if k in lin:  # Accidentely linspace falls on k
                lin = np.sort(np.random.uniform(a, b, n))
            assert a not in lin and k not in lin, (
                'a or k in lin, this is a bug. Pls report. '
                f' {k}, {n}, {a}, {b}, {lin}')
            array[i:j] = lin

        return make_unique(
            np.sort(array), random_state, mode, n_duplicates)

    
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
    
    name : str, default 'LearnedDistribution'
            The name of the instance.
            
    kwargs : all other keyword arguments accepted by sklearn.neighbors.KernelDensity.
    
    
    Methods
    -------
    
    pdf : Probability Density Function of a Gaussian Mixture
    cdf : Cumulative Density Function of a Gaussian Mixture (or its approximation)
    ppf : Approximation of a Percent Point Function of a Gaussian Mixture
    rvs : Random sample generator
    """
    
    def __init__(self, x, ravel_x=True, grid_density=int(1e4), cdf_method='fast', 
                 name='KDE', **kwargs):
        
        self.name = name
        
        if kwargs.get('kernel') not in [None, 'gaussian']:
            raise ValueError('Only gaussian kernel is supported in this wrapper.')
            
        # Fitting the KDE
        x = np.asarray(x)
        if ravel_x:
            x = x.ravel()
        self._validate_shape(x)
        self.kde = ScikitKDE(**kwargs).fit(x.reshape(-1,1))
        
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
        """
        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1)
        self._validate_shape(x)
        return np.exp(self.kde.score_samples(x.reshape(-1, 1)))   
        
    def cdf(self, k, method=None):
        """ 
        Cummulative density function of the estimated distribution.
        """
        method = method or self.cdf_method
        
        if method == 'fast':
            fast = self._cdf_fast(k)
            # Handling out of support k by computing it precisely
            # out_of_support.sum() gives the number of out_of_support values
            out_of_support = np.isnan(fast)
            if out_of_support.any():
                fast[out_of_support] = self.cdf(k[out_of_support], method='precise')
            return fast
        
        elif method == 'precise':
            data = np.asarray(self.kde.tree_.data)
            out = 0.0
            for data_point in data:
                out += norm.cdf(k, loc=data_point, scale=self.kde.bandwidth)
            return out / data.size
        
        else:
            raise NotImplementedError('Method must be one of {"fast", "precise"}')
            
    def ppf(self, q):
        """
        This method approximates the ppf based on linear interpolation
        of the cdf on self.grid_density many points. There is no formula
        for precise computation of gaussian mixture ppf. Therefore, if
        we wanted a precise function, we would need to bisect the cdf.
        Bisecting is very slow in comparison to just computing the cdf
        on a grid and using the interpolant to approximate the ppf.
        """
        q = np.asarray(q)
        if (q < 0).any() or (q > 1).any():
            raise ValueError('Value in q out of PPF support (0, 1).')
        return self._ppf(q)

    def _get_ppf(self):
        """
        In order to get a fast ppf function, we will sample
        the cdf on its support using a grid of desired density.
        Then we create an interpolant which maps the values
        inversly, thus getting an approximation to a ppf func.
        
        Theoretically, ppf(0), ppf(1) must map to -inf, +inf.
        We can not interpolate to infinite values, therefore,
        for the ppf range, we pick the closest finite values 
        from the grid. Everything outside will be set to ±inf.
        
        This way we obtain a precise enough interpolant which
        also maps the 0 and 1 to ±inf. 
        """

        def argvalid(arr):
            """
            Returns 2-tuple of indices `first` and `last`
            which can be used to slice `arr` to get only
            valid values (0+ε < valid < 1-ε). Function assumes
            `arr` is sorted and finite. ε is a multiple of 
            the tiniest number which can be represented in 
            float64. In theory ε could be 0, but we get
            numerical instabilities when interpolating later.
            If all values from left side are valid, first = 0.
            If all values from right side are valid, last = None.
            """
            v = arr > 0 #10 * np.finfo(np.float64).tiny
            w = arr < 1 #- 10 * np.finfo(np.float64).tiny
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
        
        if np.nansum(np.abs(np.array([first, last], dtype=float)))  == cdfy.size:
            raise ValueError('All grid points between a, b are invalid.')
            
        # Final x, y
        cdfx = cdfx[first:last]
        cdfy = cdfy[first:last]
        
        # New valid a, b (support of the cdf)
        self.a, self.b = cdfx[0], cdfx[-1]
        
        # Support of the ppf
        self.ppfa, self.ppfb = cdfy[0], cdfy[-1]
        
        # Real grid density (some of the points might have been excluded as invalid)
        self.grid_density = cdfx.size
        
        # Since the cdf is already evaluated, we can also just store it
        # to use it for fast cdf approximation (self.cdf(method='fast')
        self._cdf_fast = interp1d(cdfx, cdfy, bounds_error=False,
                                  fill_value=(np.nan))
        
        return interp1d(cdfy, cdfx, bounds_error=False, fill_value=(-np.inf, np.inf))
    