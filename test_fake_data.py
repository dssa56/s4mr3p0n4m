from fake_data import generate
import numpy as np

gen_args = {'t': 1000, 'maxi': 10., 'mini': 0., 'sig_depth': 2, 'sig_len': 20, 'delay': 3, 'grad': 4}

class TestGenerate():

	series = generate(**gen_args)

	def test_series_sanity(self):
		assert self.series[0] == gen_args['maxi']

	sig_start = min(np.where(series == gen_args['maxi']-gen_args['sig_depth'])[0])

	def test_sig_start(self):
		assert self.sig_start > 0
	sig_end = sig_start

	while series[sig_end] == series[sig_start]:
		sig_end += 1

	def test_sig_endpoints_sanity(self):
		assert self.sig_end > self.sig_start

	d_len = np.where(series[sig_end:] == gen_args['maxi'])[0].sum()

	def test_sig_len(self):
		assert all(np.where(self.series[self.sig_start : self.sig_start + gen_args['sig_len']]
			== gen_args['sig_len'])[0])

	def test_sig_delay(self):
		assert self.d_len == gen_args['delay'] 

	def test_grad(self):
		assert(self.series[self.sig_end + gen_args['delay']]
			- self.series[self.sig_end + gen_args['delay'] + 1]
			== gen_args['grad'])
