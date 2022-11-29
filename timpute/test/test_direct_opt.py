from ..direct_opt import *

def test_khatri_rao():
    np.random.seed(5)
    a = np.random.rand(8, 4)
    b = np.random.rand(7, 4)
    factors_1 = [a,b]
    test_1 = khatri_rao(factors_1)
    assert (test_1.shape == (7*8,4))
    assert (test_1[7*8-1,3] == a[7,3]*b[6,3])

    c = np.random.rand(6, 4)
    d = np.random.rand(5, 4)
    e = np.random.rand(4, 4)
    factors_2 = [c,d,e]
    test_2 = khatri_rao(factors_2)
    assert (test_2.shape == (6*5*4,4))
    assert (test_2[6*5*4-1,3] == c[5,3]*d[4,3]*e[3,3])

def test_factors_to_tensor():
    np.random.seed(5)
    a = np.random.rand(8, 4)
    b = np.random.rand(7, 4)
    c = np.random.rand(6, 4)
    d = np.random.rand(5, 4)
    res = factors_to_tensor([a, b, c, d])
    assert res.shape == (8, 7, 6, 5)
    assert res[1,2,3,4] == np.sum([a[1, rr] * b[2, rr] * c[3, rr] * d[4, rr] for rr in range(4)])
