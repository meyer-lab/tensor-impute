from ..direct_opt import *

def test_factors_to_tensor():
    a = np.random.rand(8, 4)
    b = np.random.rand(7, 4)
    c = np.random.rand(6, 4)
    d = np.random.rand(5, 4)
    res = factors_to_tensor([a, b, c, d])
    assert res.shape == (8, 7, 6, 5)
    assert res[1,2,3,4] == np.sum([a[1, rr] * b[2, rr] * c[3, rr] * d[4, rr] for rr in range(4)])
