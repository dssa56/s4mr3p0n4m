import numpy as np

def generate(t = 100, maxi = 20, mini = 0, sig_depth = 2, sig_len = 4, delay = 10, grad = 1):

	t_start = np.random.choice(t-sig_len-delay-1)

	series = maxi*np.ones([t_start])

	series = np.concatenate((series, np.ones([sig_len])*(maxi -sig_depth)))

	series = np.concatenate((series, np.ones([delay])*maxi))

	downturn = np.array([maxi - (i+1)*grad for i in range(t-len(series))])

	return np.concatenate((series, downturn))

