from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
from hypothesis import given

import numpy as np
import ursa.util as U


def test_triu():
    x = np.array(np.arange(9)).reshape((3,3))
    assert np.all(U.triu(x) == np.array([1,2,5]))
    
def test_pearsonr_unit():
    x = np.arange(5)
    a = 2.0
    b = -3.0
    assert np.allclose(U.pearsonr(x, x), 1.0)
    assert np.allclose(U.pearsonr(x, x * a + b), 1.0)
    
def test_pearsonr_symmetric():
    x = np.arange(3*4).reshape((3,4))
    y = np.arange(10,10+3*4).reshape((3,4))
    assert np.allclose(U.pearsonr(x, y), U.pearsonr(y, x))

@given(data=arrays(np.float64, (2,3), elements=floats(-1e+300, 1e+300, allow_nan=False, allow_infinity=False)))
def test_pairwise_diagonal(data):
    from scipy.spatial.distance import euclidean
    assert np.allclose(U.pairwise(euclidean, data, normalize=False, dtype=np.float64).diagonal(), 0.0)
    
@given(data=arrays(np.float64, (2,3), elements=floats(allow_nan=False, allow_infinity=False)))
def _test_pairwise_symmetric(data):
    from scipy.spatial.distance import euclidean
    M = U.pairwise(euclidean, data, normalize=False, dtype=np.float64)
    assert np.all(M == M.T)
    
@given(data=arrays(np.float64, (2,3), elements=floats(allow_nan=False, allow_infinity=False)))
def _test_pairwise_parallel(data):
    from scipy.spatial.distance import euclidean
    M = U.pairwise(euclidean, data, normalize=False, dtype=np.float64, parallel=False)
    L = U.pairwise(euclidean, data, normalize=False, dtype=np.float64, parallel=True)
    assert np.all(M == L)
