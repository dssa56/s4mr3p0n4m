import numpy as np


def is_weekend(t):
    b = [True if t[i] - t[i + 1] > 500 else False for i in range(len(t) - 1)]
    return True if any(b) else False


def get_samples(prices, times, n_samples, sample_len):
    inds = np.random.choice(np.arange(len(prices) - sample_len + 1),
                            [n_samples])
    accept = []
    for ind in inds:
        if not is_weekend(times[ind: ind + sample_len]):
            accept.append(prices[ind: ind + sample_len])
    return accept
