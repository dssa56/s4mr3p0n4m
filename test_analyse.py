from analyse import get_means_devs
import numpy as np
import scipy.stats as st


def test_means():
    data = np.random.choice(range(100), 1000)
    m, d = get_means_devs(data, 10)
    assert m[30] == np.mean(data[21:31])
    assert d[100] == np.sqrt(st.moment(data[91:101], moment=2))
