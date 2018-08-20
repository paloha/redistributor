# -*- coding: utf-8 -*-
from __future__ import division

import sys
import itertools
import numpy as np
from scipy.stats import norm
from sklearn.base import TransformerMixin


def save_redistributor(d, path):
    """Saves the Redistributor or Redistributor_multi object to a file."""
    import joblib
    joblib.dump(d, path)


def load_redistributor(path):
    """Loads the Redistributor or Redistributor_multi object from a file."""
    import joblib
    return joblib.load(path)


def _divisors(x, returnOne=True, returnX=True):
    """
    Generates all divisors of a number. Divisors are not yielded in order.
    By default, 1 and x are included in result.

    Parameters
    --------
    x: int
        Number for which we need to find all its divisors.

    returnOne: bool, optional, default True
        Flag specifying if number 1 should be included in results.

    returnX: bool, optional, default True
        Flag specifying if the number x itself should be included in results.

    Yields
    --------
    One int at a time, that divides x equally. No duplicates.
    """
    assert x > 0, 'Must be positive > 0'
    if x == 1:
        if returnOne or returnX:
            yield 1
            return
        else:
            return
    if returnOne:
        yield 1
    if returnX:
        yield x
    i = 2
    while i*i<=x:
        if x%i == 0:
            yield i
            if x//i != i:
                yield x//i
        i += 1


class Redistributor(TransformerMixin):
    """
    An algorithm for automatic transformation of data from arbitrary
    distribution into arbitrary distribution. Source distribution
    can be known beforehandand or learned from the data. Transformation
    is piecewise smooth, monotonic and invertible.

    Implemented as a Scikit-learn transformer. Can be fitted on 1D vector
    (for more dimensions use Redistributor_multi wrapper) and saved to be used
    later for transforming other data assuming the same source distribution.

    Uses source's and target's cdf() and ppf() to infer the
    transform and inverse transform functions.

    transform_function = target_ppf(source_cdf(x))
    inverse_transform = source_ppf(target_cdf(x))

    Time complexity
    --------
    This algorithm can scale to a large number of samples > 1e08.
    Algorithm can be sped up by:
    1. Turning off input data validation by setting validate_input=False
    2. Choosing target distribution with fast cdf and ppf functions.
    3. Specifying the source distribution if it is known.

    Parameters
    --------
    target: obj, optional, default scipy.stats.norm(loc=0, scale=1)
        Class specifying the target distribution. Must implement
        cdf(), ppf() and ideally pdf() methods.
        Continuous distributions from scipy.stats can be used.

    known_distribution: obj, optional, default None
        Object specifying the source distribution when known.
        Must implement cdf(), ppf() and ideally pdf() methods.
        Continuous distributions from scipy.stats can be used.

    bbox: tuple, optional, default None
        Tuple (a, b) which represent the lower and upper boundary
        of the source distribution's domain.

    closed_interval: bool, optional, default True
        If True, the values set in bounding box are included into
        the interpolation range.

    prevent_same: bool, optional, default True
        Flag, specifyig whether to prevent same values in the
        interpolated vector.

    tinity: int, default 1e06
        Specifies divisor of noise added to vectors with same elements.
        Bigger the tinity, smaller the noise.

    validate_input: bool, optional, default True
        Flag specifying if the input data should be validated.
        Turn off to save computation time. Output is undefined
        if the data are not valid.

    bins: int, optional, default 0
        Number of bins which are used to infer the latice density
        when the distribution is learned. Irelevant with known_distribution.

        If bins = x.size, then latice density = 100%. The interpolation
        step = x.size // bins = 1. This is most precise. If x.size is big enough,
        we can lower the latice density and the fit will remain very precise.

        If bins = 0, it is set automatically. The latice density = 100%
        is then used up to x.size = 5000. If x.size > 5000, bins = 5000.

    Attributes
    --------
    self.target_cdf: callable, default scipy.stats.norm(loc=0, scale=1).cdf
        Cummulative Density Function of target distribution.

    self.target_ppf: callable, default scipy.stats.norm(loc=0, scale=1).ppf
        Percent Point Function (inverse of cdf) of target distribution.

    self.source_cdf: callable, default None
        Either known_distribution.cdf function of source distribution
        or learned on fit() by interpolation on latice.

    self.source_ppf: callable, default None
        Either known_distribution.ppf function of source distribution
        or learned on fit() by interpolation on latice.

    self.a: float, default None
        Beginning of the interval of the source distribution domain.

    self.b: float, default None
        End of the interval of the source distribution domain.

    self.n: int, default None
        Size of the training data extened by bounding box borders if necessary.

    self.fitted: bool, default False
        Flag that specifies whether the fit() was already called.

    Limitations
    --------
    - Output is undefined when most of the data points have the exact
      same value, therefore data should not be sparse - multiple zeros etc.
      Algorithm handles small amounts of the exact same values by adding tiny noise.

    - Plotting of learned pdf() is just a smoothed approximation obtained
      by 1st derivative of the learned piece-wise smooth cdf() and it
      serves only as a visual aid.

    Examples
    --------
    TODO

    """

    def __init__(self,
                 target=norm(loc=0, scale=1),
                 known_distribution=None,
                 bbox=None,
                 closed_interval=True,
                 prevent_same=True,
                 tinity=int(1e06),
                 validate_input=True,
                 bins=0):

        self.known_distribution = known_distribution
        self.bbox = bbox
        self.closed_interval = closed_interval
        self.prevent_same = prevent_same
        self.tinity = tinity
        self.validate_input = validate_input

        self.bins = bins # Might chang on fit()
        self.target_cdf = target.cdf
        self.target_ppf = target.ppf

        # Will be computed or changed on fit()
        self.source_cdf = None
        self.source_ppf = None
        self.a = None
        self.b = None
        self.n = None
        self.fitted = False

        if self.known_distribution is not None:
            self.fit()

    def fit(self, x=None):
        """
        Calls all necessary methods to infer cdf and ppf of the source distribution.
        Source distribution can be either specified directly by its function or learned
        from the training data on a closed or opened interval. Learning approximates
        the cdf and ppf by interpolating on a latice which density is infered from bins.

        Parameters
        --------

        x: 1D numpy array, optional, default None
            Training data from which the source distribution is learned.
            Must be specified if self.known_distribution is None.
        """

        if self.known_distribution is not None:
            assert self.bbox is not None, 'Bounding box must be specified when using known_distrubition.'
            self._infer_a_b(x, self.bbox)
            try:
                self.source_cdf = self.known_distribution.cdf
                self.source_ppf = self.known_distribution.ppf
            except AttributeError:
                print('Class known_distribution must implement cdf(), ppf() and ideally pdf() methods.', file=sys.stderr)
                raise
            try:
                self.source_pdf = self.known_distribution.pdf
            except AttributeError:
                self.source_pdf = None
            self.known_distribution = self.known_distribution
        else:
            assert x is not None, 'If known_distribution is None, X must be specified.'

            # Calculate bounding box
            self._infer_a_b(x, self.bbox)

            # Validate input data
            if self.validate_input: self._validate_input(x)

            # Compute number of bins to use
            self._infer_nbins(x.size, self.bins)

            # Ensure, that a and b values are in x
            x = self._enforce_borders(x, self.closed_interval)

            # Get the size of x with borders a and b
            self.n = x.size

            # Learn the source distribution
            self._infer_cdf_ppf(x, self.prevent_same)

        self.fitted = True

    def transform(self, x):
        """
        Applies learned transformation function to the data x.

        Parameters
        --------
        x : numpy array
            Data to transform. Must be within self.a, self.b interval.

        Returns
        --------
        Numpy array of transformed data.
        """

        if self.validate_input: self._validate_input(x)
        return self.target_ppf(self.source_cdf(x))

    def inverse_transform(self, x):
        """
        Applies learned inverse transform function to the data x.

        Parameters
        --------
        x : numpy array
            Data to inverse transform. Must be within the interpolation interval
            of learned transform function.

        Returns
        --------
        Numpy array of inverse transformed data.
        """

        return self.source_ppf(self.target_cdf(x))

    def _validate_input(self, data):
        """
        Validation of the input data used to validate inputs to fit() and transform().
        Can be turned off globally by setting self.validate_input = False.

        Parameters
        --------
        data: numpy array of data to be validated
        """
        assert data.dtype == np.float64 or data.dtype == np.float32, 'Data must be float64 of 32.'
        assert data.ndim == 1, 'Data must be stored in 1D numpy array. You can use np.ravel(data).'
        assert all(np.isfinite(data)), 'Data must not contain np.nan or np.inf.'
        assert self.a <= np.min(data) and np.max(data) <= self.b, 'Data out of interval set by bbox.'

    def _infer_a_b(self, data, bbox):
        """
        Infers self.a and self.b which represent the lower and upper boundary
        of the source distribution's domain.

        Parameters
        --------
        data: numpy array or None
            Training data used to infer the a and b automaticaly by taking min and max.
            Can't be None if the bbox is None.

        bbox: tuple or None
            Tuple (a, b) which represent the lower and upper boundary
            of the source distribution's domain.
        """

        if bbox is None: bbox = (np.min(data), np.max(data))
        assert len(bbox) == 2, 'Please specify just two values in bbox.'
        assert bbox[0] < bbox[1], 'First element of bbox must be smaller than the second.'
        self.a = bbox[0]
        self.b = bbox[1]

    def _enforce_borders(self, x, closed_interval):
        """
        Inserts self.a and self.b into the array x on the first and
        last positions respectively. If closed_interval=True, the
        values are extended by small amount, so the actual values
        a and b are still included in the interpolation range.

        Parameters
        --------
        x: numpy array
            1D vector into which the values are inserted.

        closed_interval: bool
            Flag, specifying the opened or closed interval.
        """
        xmin = np.min(x)
        xmax = np.max(x)
        extend = (xmax-xmin) / x.size / 100 if closed_interval else 0
        if xmin != self.a-extend: x = np.insert(x, 0, self.a-extend)
        if xmax != self.b+extend: x = np.append(x, self.b+extend)
        return x

    def _infer_nbins(self, n, bins):
        """
        Infers or validates the number of bins.

        Parameters
        --------
        n: int
            Size of the training data.

        bins:
            User specified value of bins.
        """
        if bins == 0: bins = min(n, 5000)
        assert bins > 2 and bins <= n
        self.bins = bins

    def _prevent_same(self, x):
        """
        Adds tiny noise to prevent same values in a vector if there are any.

        Parameters
        --------
        x: 1D numpy array
            Array with potential of having not unique elements.

        Returns
        --------
        Sorted array with added small noise based on self.tinity.
        The aim is to have all elements in the array unique.
        """
        s = np.array(list(set(x)))
        if x.size != len(s):
            # Find the smallest distance between two consecutive elements
            mindist = np.abs(np.min(np.ediff1d(s)))

            # Add tiny noise that is smaller than the smalledst distance
            tiny_noise = np.random.rand(x.size) * (mindist / self.tinity)
            x += tiny_noise
            x = np.sort(x)
        return x

    def _infer_cdf_ppf(self, x, prevent_same):
        """
        Learning of the target distribution is done by linear interpolation
        on a latice. This method infers approximation of source's
        Cumulative distribution function and Percent point function.

        Parameters
        --------
        x: numpy array
            1D vector used for the interpolation.
        prevent_same: bool
            Flag, specifying whether to add small noise to
            values on the latice so the interpolation data
            do not have the exact same values. Applied only
            if there are some.
        """

        # Define latice on which to interpolate (more points = bigger precision)
        latice = np.linspace(0, self.n-1, self.bins).astype(int)

        # Get values of x on latice points by partial sort
        x.partition(latice)
        values_on_latice = np.sort(x[latice])

        # Adds tiny noise to prevent same values if necessary
        if prevent_same: values_on_latice = self._prevent_same(values_on_latice)

        # Normalize the latice by nubmer of samples
        # Cummulative density function must sum up to 1
        latice = latice / (self.n-1)

        # Interpolate to get cdf and ppf
        from scipy.interpolate import interp1d
        self.source_cdf = interp1d(values_on_latice, latice, kind='linear')
        self.source_ppf = interp1d(latice, values_on_latice, kind='linear')

    def compute_empirical_cdf_error(self, x, error_func='mse'):
        """
        Computes error caused by approximation of transform function.
        Error computation needs to sort all data. It can take a long time for big arrays.

        Parameters
        --------
        x: numpy array
            1D vector of transformed data.
        error_func: callable or one of {'mae', 'mse'}
            Callable that computes error on two vectors.
            'mae' = Error in L1 norm (Mean Absolute Error)
            'mse' = Error in L2 norm (Mean Squared Error)

        Returns
        --------
        Float value of specified error or return value of callable.
        """

        if error_func == 'mse':
            from sklearn.metrics import mean_squared_error as errfunc
        elif error_func == 'mae':
            from sklearn.metrics import mean_absolute_error as errfunc
        else:
            errfunc = error_func
            assert callable(errfunc), 'Set error_func to "mse", "mae" or function that computes error.'

        transformed = self.transform(x)
        transformed = np.sort(transformed)
        n = transformed.size
        p_lower = self.target_cdf(transformed[0])
        p_upper = self.target_cdf(transformed[-1])
        v = p_lower + ((p_upper - p_lower)/(n-1)) * np.arange(0,n)
        w = self.target_cdf(transformed)
        return errfunc(v, w)

    def plot_transform_function(self, figsize=(15,2)):
        """
        Displays matplotlib plot of the fitted transform function.

        Parameters
        --------
        figsize : tuple (width, height), optional, default (15,2)
            Desired size of the figure.
        """
        assert self.fitted, 'First, the object must be fitted using fit().'

        x_axis = np.linspace(self.a, self.b, 1000)
        transformed = self.transform(x_axis)

        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(x_axis, transformed)
        plt.title('Learned monotonic piecewise smooth transform function')
        plt.xlim(self.a, self.b)
        plt.show()
        plt.close()

    def plot_source_pdf(self, smoothed=True, figsize=(15,2)):
        """
        Displays matplotlib plot of the learned source probability density function.

        Parameters
        --------
        smoothed: boolean, optional, default True
            Applies savgol_filter to smooth the curve that is disturbed by derivatives at bin transitions.

        figsize : tuple (width, height), optional, default (15,2)
            Desired size of the figure.
        """
        assert self.fitted, 'First, the object must be fitted using fit().'

        steps = 1000
        step = (self.b - self.a) / steps

        x_axis = np.linspace(self.a, self.b, steps, endpoint=False)
        x_axis = x_axis[1:]

        if self.known_distribution:
            if self.source_pdf is not None:
                curve = self.source_pdf(x_axis)
                title = 'Pdf of known distribution.'
                text = None
            else:
                print("Can't plot source pdf. Class of known distribution does not implement pdf().", file=sys.stderr)
                return
        else:
            from scipy.misc import derivative
            curve = derivative(self.source_cdf, x_axis, dx=step / 2)
            sm = ''
            if smoothed:
                from scipy.signal import savgol_filter
                win_width = steps//8
                if win_width % 2 == 0: win_width1 += 1
                curve = savgol_filter(curve, 51, 1)
                sm = ' (smoothed)'
            title = 'Source_pdf{} approximated using {} bins.'.format(sm, self.bins)
            text = '\nMay not show appropriate results with certain source distributions.'
            text += ' In that case rather plot a histogram of your train data.'

        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(x_axis, curve)
        plt.title(title)
        plt.xlim(self.a, self.b)
        plt.ylim((0., 1.2*np.max(curve)))
        if text is not None:
            ax = plt.gca()
            plt.text(0.5, -0.4,
                     text,
                     size=10,
                     ha='center',
                     va='bottom',
                     transform=ax.transAxes)
        plt.show()
        plt.close()

    def plot_hist(self, data, nbins=None, title='Histogram', figsize=(15,2), xlim=None):
        """
        Displays matplotlib histogram of the specified data and nbins.

        Parameters
        --------
        data : numpy array
            Specifying the position of data points on x axis.

        nbins : int, optional, default self.bins
            Specifying the number of bins of the histogram.
            If None, the number will be automatically set by matplotlib.

        title : str, optional, default 'Histogram'
            Title of the plot.

        figsize : tuple (width, height), optional, default (15,2)
            Desired size of the figure.

        xlmi : float, optional, default None
            Limit of x axis.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        if self.known_distribution is None:
            if nbins is None and self.n > 2500: nbins = min(self.bins, 301)
        if xlim is not None:
            plt.xlim(xlim)
        plt.hist(data, nbins)
        plt.title(title)
        plt.show()
        plt.close()


class Redistributor_multi():
    """
    Multi-dimensional wrapper for Redistributor.
    Allows to use multiple Redistributors on equal-sized slices (submatrices) of
    N-dimensional input array. Utilizes parallel processing with low memory footprint.

    Parameters:
    --------

    redistributors: numpy array of Redistributor objects
        Objects that will be used to operate on the data within each slice
        defined by nsub. Shape must be the same as nsub.

    nsub: tuple of int
        Tuple specifying how to split multidimensional input
        into submatrices. Corresponding redistributor object will
        be applied on each submatrix separately. Each int
        specifies to how many equal submatrices corresponding
        axis should be split. There must be exactly one int
        for each axis of multidimensional input data.

    cpus: int >= 0
        Number of cpu cores to use in multiprocessing.
        If 0, all cores will be used.
    """

    def __init__(self, redistributors, nsub, cpus=0):
        self.nsub = np.array(nsub)
        if np.prod(self.nsub == 1):
            print('WARNING: Using Redistributor_multi on whole matrix is slower than just Redistributor.', file=sys.stderr)
        self.redistributors = redistributors
        assert self.redistributors.size == np.prod(self.nsub), 'Specify one redistributor per slice.'
        self.cpus = self._infer_cpus(cpus)
        self.fitted = False

    def _infer_cpus(self, cpus):
        """
        Keeps the defined number of cpus or tries to set it to all
        available cpus if cpus = 0.
        """
        if cpus == 0:
            try:
                from multiprocessing import cpu_count
                return cpu_count()
            except NotImplementedError:
                return 1  # default
        else:
            return cpus

    def fit(self, x=None, size_limit=0):
        """
        Fits all Redistributor objects in self.redistributors.

        Parameters
        --------
        x : numpy array, optional, default None
            Data to fit on. If None, Redistributors must have
            source set explicitly.

        size_limit : float, optional, default 0
            Should be 3x smaller than available memory after loading the data in GB.
            Based on this value, the appropriate size of chunk is infered.
            If size_limit == 0, the best value is computed automatically.
        """
        if x is not None:
            self.x = x
            self.machinery('fit', size_limit=size_limit)
        self.fitted = True

    def transform(self, x, inplace=True, size_limit=0):
        """
        Transforms data in x using fitted Redistributors.

        Parameters
        --------
        x : numpy array
            Data to transform.

        inplace : bool, optional, default True
            Flag specifying whether to change the data in place without
            keeping the original values stored in x or operate on copy.

        size_limit : float, optional, default 0
            See docstring of self.fit

        Returns
        --------
        Transformed data with same shape as input.
        """
        self.inplace = inplace
        self.x = x if self.inplace else x.copy()
        return self.machinery('transform', size_limit=size_limit)

    def inverse_transform(self, x, inplace=True, size_limit=0):
        """
        Inverse transforms the data in x using fitted Redistributors.

        Parameters
        --------
        x : numpy array
            Data to transform.

        inplace : bool, optional, default True
            Flag specifying whether to change the data in place without
            keeping the original values stored in x or operate on copy.

        size_limit : float, optional, default 0
            See docstring of self.fit

        Returns
        --------
        Inverse transformed data with same shape as input.
        """
        self.inplace = inplace
        self.x = x if self.inplace else x.copy()
        return self.machinery('inverse', size_limit=size_limit)

    def _locate_subarrays(self, xshape, nsub):
        """
        Locates subarrays within matrix x according to nsub.

        Returns
        --------
        - Numpy array of indices that locate the subarrays within matrix x.
        - Shape of subarray that would be produced by slicing the matrix.
        Each subarray is of equal shape.
        """

        nsub = np.array(nsub)
        shape = np.array(xshape)
        steps, rems = np.divmod(shape, nsub)
        assert all(rems == 0), 'Nsub does not divide the x equally on {} axes.'.format(np.nonzero(rems)[0])
        steps = np.array(shape / nsub).astype(int)
        output_shape = steps

        subarray_indices = [list(zip(range(0,shape[axis],steps[axis]),
            range(0+steps[axis], shape[axis]+steps[axis], steps[axis])))
                for axis in range(len(nsub))]

        return np.array(list(itertools.product(*subarray_indices))), output_shape

    def _indices_to_slices(self, indices):
        """
        Converts numpy array of start and stop indices to numpy array of slices.
        """
        shape = indices.shape
        assert shape[-1] == 2, 'Last axis of indices must contain 2 or 3 elements, start, stop and step.'
        indices = indices.reshape(-1, 2)
        return np.array([slice(*ind) for ind in indices]).reshape(shape[:-1])

    def _get_size_limit(self):
        """
        Checks available memory and decides on size_limit for self._get_chunksize.
        """
        try:
            import psutil
            size_limit = psutil.virtual_memory().available / 2.2e09
            if size_limit < 0.1:
                print('''WARNING: It seems you have too low available memory.
                The speed might be influenced significantly. For optimal speed it
                is good to have ~2x size_limit of free memory after loading tha data
                that are being processed. Size_limit for one chunk of sharedctypes
                array was set to default 0.5GB. Consider using self.cpus = 1.''', file=sys.stderr)
                return 0.5
            else:
                return size_limit
        except:
            import sys
            print('''WARNING: Unable to obtain the size of available memory.
            Size_limit for one chunk of sharedctypes array was set to default 0.5GB.
            For optimal speed it is good to have ~2x size_limit of free memory
            after loading tha data that are being processed.''', file=sys.stderr)
            return 0.5

    def _get_chunksize(self, size_limit):
        """
        Returns number of slices that should be in one chunk
        so the size of chunk is ideally equal to size_limit.
        Divides all slices to chunks with agreement to self.nsub.
        The size of chunk will be bigger if it is not possible.
        No matter the size, it never returns 1, because the
        whole class looses its meaning. Returns 1 only if the
        np.prod(self.nsub) == 1 which is discouraged.

        Parameters
        --------
        size_limit : float
            Ideal size of one chunk so the sharedctypes array used
            in multiprocessing pipe in self.machinery is created as fast
            as possible.
        """
        all_slices = np.prod(self.nsub)
        last_best = 1
        slice_size = self.x.nbytes / all_slices * 1e-09 # in GB
        for i, n in enumerate(self.nsub):
            if n == 1:
                continue
            returnOne = False if np.prod(self.nsub[i+1:]) == 1 else True
            possible_steps_on_axis = np.array(list(reversed(sorted(_divisors(n, returnOne=returnOne, returnX=True)))))
            slices_in_chunk = possible_steps_on_axis * np.prod(self.nsub[i+1:])
            for s in slices_in_chunk:
                if s * slice_size <= size_limit:
                    return s
                last_best = s
        return last_best

    @staticmethod
    def init_global_array(array):
        """Initializer of shared array for multiprocessing pool."""
        global arr
        arr = array

    @staticmethod
    def populate(args):
        """
        Static method called by child processes that applies desired
        function of redistributor on the data from shared matrix on
        desired location and populates the result back to the shared matrix.
        """
        index, location, shape, redistributor, purpose, cpus = args
        if cpus == 1:
            # Get the access to the global array
            matrix = arr
        else:
            # Get the access to shraed array
            matrix = np.ctypeslib.as_array(arr)

        # Take the vector from the shared array
        v = matrix[tuple(location)].ravel()

        if purpose == 'fit':
            redistributor.fit(v)
        elif purpose == 'transform':
            matrix[tuple(location)] = redistributor.transform(v).reshape(shape)
        elif purpose == 'inverse':
            matrix[tuple(location)] = redistributor.inverse_transform(v).reshape(shape)

        return index, redistributor

    def machinery(self, purpose, size_limit):
        """
        Handles locating subarrays and their parallel processing in chunks.

        Parameters
        --------
        purpose : one of {'fit', 'transform', 'inverse'}
            Specifies what should be done with the data.

        size_limit : float
            Ideal size of one chunk. 0 = automatic.
        """

        # Number of all subarrays that will be used
        n_subarrays = np.prod(self.nsub)

        # Get subarray locations (list of indices)
        indices, output_shape = self._locate_subarrays(self.x.shape, self.nsub)

        # Avoiding multiprocessing and the overhead of creating shared_array
        if self.cpus == 1:
            locations = self._indices_to_slices(indices).tolist()
            Redistributor_multi.init_global_array(self.x)
            list(map(Redistributor_multi.populate,
                    zip(range(len(locations)),
                        locations,
                        itertools.repeat(output_shape),
                        self.redistributors.ravel(),
                        itertools.repeat(purpose),
                        itertools.repeat(self.cpus))))

        # Using pool of child processes running in parallel
        else:
            from multiprocessing import Pool
            from multiprocessing import RawArray

            # Get chunksize for splitting the self.x so creation of shared_array is faster
            if size_limit == 0:
                size_limit = self._get_size_limit()
            chunksize = self._get_chunksize(size_limit)
            stepsize = n_subarrays // chunksize

            chunked_indices = indices.reshape(stepsize, -1, len(self.nsub), 2)
            locations_of_subarrays_within_each_chunk = self._indices_to_slices(chunked_indices[0]).tolist()

            for i, chunks_subarray_indices in enumerate(chunked_indices):
                # Find start and stop of chunk slice in each axis
                mins = np.min(chunks_subarray_indices, axis=2)[0]
                maxs = np.max(chunks_subarray_indices, axis=2)[-1]
                indices_of_chunk = np.array(list(zip(mins, maxs)))
                location_of_chunk = self._indices_to_slices(indices_of_chunk).tolist()

                # Create shared memory array that is accessible by child processes
                s = self.x[tuple(location_of_chunk)].copy()
                tmp = np.ctypeslib.as_ctypes(s)

                # Creating a shared array if much faster when the underlying C code
                # can make a copy of it. If there is not that much available memory left
                # it replaces value by value in place which takes significantly more time.
                # That is the reason this is chunkized into smaller arrays and done in a for loop.
                shared_array = RawArray(tmp._type_, tmp)
                del s
                del tmp

                pool = Pool(processes=self.cpus,
                                  initializer=Redistributor_multi.init_global_array,
                                  initargs=(shared_array, ))

                p = pool.map(Redistributor_multi.populate,
                              zip(range(i*chunksize, (i+1)*chunksize),
                                  locations_of_subarrays_within_each_chunk,
                                  itertools.repeat(output_shape),
                                  self.redistributors.ravel(),
                                  itertools.repeat(purpose),
                                  itertools.repeat(self.cpus)))

                # Update redistributor objects after being changed
                [np.put(self.redistributors, i, instance) for i, instance in p]

                pool.close()
                pool.join()

                if purpose != 'fit':
                    self.x[tuple(location_of_chunk)] = np.ctypeslib.as_array(shared_array)

                # Freeing the memory
                del shared_array

        # Returning the results
        if purpose == 'fit':
            output = None
        else:
            output = self.x

        # Cleaning up the object
        del self.x
        return output
