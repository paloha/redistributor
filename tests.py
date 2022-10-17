# Run tests with `pytest tests.py`
import math
import pytest
import numpy as np
from redistributor import Redistributor as R
from redistributor import KernelDensity as KDE
from redistributor import LearnedDistribution as L


# LearnedDistribution ###################################################################
# Cases #################################################################################

data_descriptions = [
    # Each entry should result in at least x of size 6 because 
    # further down downsampling with step 2 is tested
    np.array([0, 1, 2, 3, 4, 5]).astype(float),
    np.array([-2, 1, 3, -4, 4, 5]).astype(float),
    
    # Really small amount of samples
    (np.random.uniform, 0, 1, 10),
    (np.random.uniform, -5, 5, 10),
    (np.random.normal, 0, 1, 10),
    
    (np.random.uniform, 0, 1, 100),
    (np.random.uniform, -5, 5, 100),
    (np.random.normal, 0, 1, 100),
    (np.random.normal, -1000, 5000, 100),
    
    # More than 5k samples
    (np.random.uniform, 0, 1, 6000),
    (np.random.uniform, -5, 5, 6000),
    (np.random.normal, 0, 1, 6000),

    # With duplicate values
    np.repeat(np.random.uniform(-5, 5, 10), 3),
    np.repeat(np.random.uniform(-5, 5, 50), 3),
    np.repeat(np.random.uniform(-5, 5, 2000), 3),
]

bound_descriptions = [
    {'a': None, 'b': None},
    {'a': '<min', 'b': None},
    {'a': None, 'b': '>max'},
    {'a': '<min', 'b': '>max'},
]

bound_descriptions_invalid = [
    {'a': 'min', 'b': None},
    {'a': None, 'b': 'max'},
    {'a': 'min', 'b': 'max'},
]

configs = [
    {'subsample_x': None, 'fill_value': 'auto', 'resolve_duplicates': ('max', 'clip')},
    {'subsample_x': None, 'fill_value': 'auto', 'resolve_duplicates': (1e-01, 'clip')},
    {'subsample_x': None, 'fill_value': 'auto', 'resolve_duplicates': None},
    {'subsample_x': 2, 'fill_value': 'auto', 'resolve_duplicates': ('max', 'clip')},
    {'subsample_x': 2, 'fill_value': 'auto', 'resolve_duplicates': (1e-01, 'clip')},
    {'subsample_x': 2, 'fill_value': 'auto', 'resolve_duplicates': None},

]

def data_factory(description):
    """
    Produces data from the description
    """
    if isinstance(description, np.ndarray):
        data = description
    elif isinstance(description, tuple):
        func, loc, scale, size = description
        data = func(loc, scale, size)
    else:
        raise NotImplementedError()
    return data, data.min(), data.max(), data.size

def bound_factory(description, Dmin=None, Dmax=None):
    """
    Replaces bound values specified by keywords
    """
    desc = description.copy()
    eps = 1e-2
    m = {'min': Dmin, 'max': Dmax, '<min': Dmin - eps, '>max': Dmax + eps}
    for k, v in desc.items():
        if v in m:
            desc[k] = m[v]
    return desc['a'], desc['b']

def l_factory(data_description, bounds_description, config):
    """
    Takes in the LearnedDistribution context descriptions
    and returns an initialized object
    """
    D, Dmin, Dmax, Dsize = data_factory(data_description)
    a, b = bound_factory(bounds_description, Dmin, Dmax)
    return L(D, a, b, **config)

# Tests ##################################################################################   

@pytest.mark.parametrize('bounds', bound_descriptions_invalid)
def test_l_init_invalid(bounds):
    D = np.array([1., 2., 3.])
    a, b = bound_factory(bounds, D.min(), D.max())
    # a must be < than xmin and b > x.max
    with pytest.raises(ValueError) as e_info:
        l = L(D, a, b)

@pytest.mark.parametrize('data_description', data_descriptions)
@pytest.mark.parametrize('bounds_description', bound_descriptions)
@pytest.mark.parametrize('config', configs)
def test_l_init(data_description, bounds_description, config):
    # This function could probably be split into separate tests
    # using parametrized yielding fixtures. It would count each
    # check as a separate test and it should reuse the objects
   
    try: # Initialization should not raise an exception
        l = l_factory(data_description, bounds_description, config)
    except Exception as e:
        assert False, e
        
    # Checks for cdf
    check_l_cdf_bounds(l)
    check_l_cdf_edges(l)
    check_l_cdf_inversion(l)
    check_l_cdf_array_inversion(l)
    check_l_cdf_values(l)
    
    # Checks for ppf
    check_l_ppf_bounds(l)
    check_l_ppf_edges(l)
    check_l_ppf_inversion(l)
    check_l_ppf_array_inversion(l)
    check_l_ppf_values(l)
    
    # Checks for rvs
    check_l_rvs_values(l)
    
def check_l_cdf_inversion(l):
    a, b = l._get_support()
    assert np.array_equal(l.ppf(l.cdf([a, b])), [a, b])
    
def check_l_cdf_array_inversion(l):
    n_samples = 100
    q = np.linspace(*l._get_support(), n_samples)
    p = l.cdf(q)
    assert np.allclose(l.ppf(p), q)
    
def check_l_cdf_bounds(l):
    if l.resolve_duplicates is None and l.a is None:
        # In this case we expect to fail the assert but the user is
        # warned that xmin occurs multiple times which changes the
        # mapping of the interp1d
        return
    else:
        assert np.array_equal(l.cdf(l._get_support()), l._get_support_ppf())
    
def check_l_cdf_edges(l):
    if l.resolve_duplicates is None:
        # In this case we expect to fail the assert but the user is
        # warned that xmin occurs multiple times which changes the
        # mapping of the interp1d
        return
    else:
        delta = 1 / (l.bins + 1)
        assert np.allclose(l.cdf([l.xmin, l.xmax]), [delta, 1 - delta])
    
def check_l_cdf_values(l):
    n_samples = 100
    a, b = l._get_support()
    q = np.linspace(a, b, n_samples)
    errors = []
    p = l.cdf(q)
    
    if p.min() < 0 or p.max() > 1:
        errors.append('CDF output out of support.')
    if not np.isfinite(p).all():
        errors.append('CDF output contains non-finite values.')
    if not p.size == n_samples:
        errors.append(f'CDF output contains wrong amount of samples.')
    assert not errors, 'Errors occured.' 
    
def check_l_ppf_bounds(l):
    assert np.array_equal(l.ppf(l._get_support_ppf()), l._get_support())
    
def check_l_ppf_inversion(l):
    if l.resolve_duplicates is None and l.a is None:
        # In this case we expect to fail the assert because
        # xmin or xmax can occur multiple times which changes
        # the mapping of interp1d
        return
    else:
        assert np.array_equal(l.cdf(l.ppf(l._get_support_ppf())), l._get_support_ppf())
        
def check_l_ppf_array_inversion(l):
    if l.resolve_duplicates is None:
        return  # Expected to be non-invertible 
    n_samples = 100
    p = np.linspace(*l._get_support_ppf(), n_samples)
    q = l.ppf(p)
    assert np.allclose(l.cdf(q), p)
    
def check_l_ppf_edges(l):
    delta = 1 / (l.bins + 1)
    assert np.allclose(l.ppf([delta, 1 - delta]), [l.xmin, l.xmax])
    assert np.allclose(l.ppf([0, 1]), l._get_support())
    
def check_l_ppf_values(l):
    n_samples = 100
    a, b = l._get_support()
    ppfa, ppfb = l._get_support_ppf()
    p = np.linspace(ppfa, ppfb, n_samples)
    errors = []
    q = l.ppf(p)
    
    if q.min() < a or q.max() > b:
        errors.append(f'PPF output out of support.')
    if not np.isfinite(q).all():
        errors.append(f'PPF output contains non-finite values.')
    if not q.size == n_samples:
        errors.append(f'PPF output contains wrong amount of samples.')
    assert not errors, 'Errors occured.' 
    
def check_l_rvs_values(l):
    n_samples = 100
    s = l.rvs(n_samples, random_state=1)
    a, b = l._get_support()
    errors = []
    if s.min() < a or s.max() > b:
        errors.append('Random sample out of cdf support.')
    if not np.isfinite(s).all():
        errors.append('Random sample contains non-finite values.')
    if not s.size == n_samples:
        errors.append('Random sample generated wrong amount of samples.')
    assert not errors, 'Errors occured.'

    
## KernelDensity #########################################################################
# Cases ##################################################################################

kde_cases = { # KDE object configs
    1: [{'low': 0,'high': 1, 'size': 2}, {'bandwidth': 0.1, 'grid_density': int(1e2)}],
    2: [{'low': 50,'high': 100, 'size': 200}, {'bandwidth': 0.1, 'grid_density': int(1e2)}],
    3: [{'low': -1,'high': 1, 'size': 5000}, {'bandwidth': 1.2, 'grid_density': int(1e4)}],
}
kdes = [KDE(np.random.uniform(**kde_cases[n][0]), **kde_cases[n][1], name=f'Case_{n}') for n in kde_cases]


# Tests ##################################################################################

@pytest.mark.parametrize("kde", kdes)
def test_kde_case_counter(kde):
    print(kde.name, kde.b)

@pytest.mark.parametrize("kde", kdes)
def test_kde_cdf_inversion(kde):
    a, b = kde._get_support()
    assert np.array_equal(kde.ppf(kde.cdf([a, b])), [a, b])

@pytest.mark.parametrize("kde", kdes)
def test_kde_ppf_inversion(kde):
    ppfa, ppfb = kde._get_support_ppf()
    assert np.array_equal(kde.cdf(kde.ppf([ppfa, ppfb])), [ppfa, ppfb])
    
@pytest.mark.parametrize("kde", kdes)
def test_kde_cdf_values(kde):
    n_samples = 100
    a, b = kde._get_support()
    k = np.linspace(a, b, n_samples)
    errors = []
    for method in ['precise', 'fast']:
        cdf = kde.cdf(k, method)
        if cdf.min() < 0 or cdf.max() > 1:
            errors.append(f'CDF {method} output out of support.')
        if not np.isfinite(cdf).all():
            errors.append(f'CDF {method} output contains non-finite values.')
        if not cdf.size == n_samples:
            errors.append(f'CDF {method} output contains wrong amount of samples.')
    assert not errors, 'Errors occured.' 
    
@pytest.mark.parametrize("kde", kdes)
def test_kde_ppf_values(kde):
    n_samples = 100
    ppfa, ppfb = kde._get_support_ppf()
    q = np.linspace(ppfa, ppfb, n_samples)
    errors = []
    ppf = kde.ppf(q)
    if kde.ppf(0) != -np.inf:
        errors.append('PPF(0) maps to wrong value.')
    if kde.ppf(1) != np.inf:
        errors.append('PPF(1) maps to wrong value.')
    if ppf.min() < kde.a or ppf.max() > kde.b:
        errors.append(f'PPF output out of support.')
    if not np.isfinite(ppf).all():
        errors.append(f'PPF output contains non-finite values.')
    if not ppf.size == n_samples:
        errors.append(f'PPF output contains wrong amount of samples.')
    assert not errors, 'Errors occured.' 
    
@pytest.mark.parametrize("kde", kdes)
def test_kde_rvs_values(kde):
    n_samples = 100
    rvs = kde.rvs(n_samples, random_state=1)
    errors = []
    if rvs.min() < kde.a or rvs.max() > kde.b:
        errors.append('Random sample out of cdf support.')
    if not np.isfinite(rvs).all():
        errors.append('Random sample contains non-finite values.')
    if not rvs.size == n_samples:
        errors.append('Random sample generated wrong amount of samples.')
    assert not errors, 'Errors occured.'

    
# OLD CASES CODE WITH METAFIXTURES #########################################################

# cases = { # KDE object configs
#     1: [{'low': 0,'high': 1, 'size': 2}, {'bandwidth': 0.1, 'grid_density': int(1e2)}],
#     2: [{'low': -10,'high': 10, 'size': 200}, {'bandwidth': 0.1, 'grid_density': int(1e2)}]
# }

# def casegen():
#     for n in cases:
#         cases[n][1].update({'name': f'Case_{n}'})
#         yield cases[n]
        
# case = casegen()

# # Meta fixture - register all fixtures in the params bellow
# @pytest.fixture(params=['kde1', 'kde2'])
# def kde(request):
#     return request.getfixturevalue(request.param)

# @pytest.fixture(scope="module")
# def kde1():
#     c = next(case)
#     return KDE(np.random.uniform(**c[0]), **c[1])

# @pytest.fixture(scope="module")
# def kde2():
#     c = next(case)
#     return KDE(np.random.uniform(**c[0]), **c[1])