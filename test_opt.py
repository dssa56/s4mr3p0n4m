import pyximport
pyximport.install(reload_support=True)
import opt
import numpy as np


def test_get_ind():
    assert (
        opt.wrap_get_ind(np.array([1, 2, 1, 4, 5, 3, 6]).astype(np.int32),
                         np.array([12, 2, 4, 2, 5, 3, 5]).astype(np.int32))
        == (2 * 4 * 5 * 3 * 6 * 12 + 4 * 5 * 3 * 6 * (2 + 4) + 5 * 3 * 6 * 2
            + 3 * 6 * 5 + 6 * 3 + 5)
    )


def test_best():
    l = np.array([[1, 2, 3, 5], [2, 3, 10, 2.4], [1, 4.5, 6, 3]]
                 ).astype(np.float64)
    assert all(opt.get_best(l) == np.array([3, 10, 6]))
