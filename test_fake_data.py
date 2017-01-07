from fake_data import generate
import numpy as np

gen_args = {'t': 1000, 'maxi': 10, 'mimi': 0, 'sig_depth': 2, 'sig_len': 20, 'delay': 3, 'grad': 4}

class TestGenerate():

	series = generate(**gen_args)

	sig_start = min(np.where(series == gen_args['mini']))

	sig_end = sig_start

	while series[sig_end] == series[sig_start]:
		sig_end += 1

	d_len = np.where(series[sig_end+1:] == gen_args['maxi']).sum()


	def test_sig_len(self):
		assert all(self.series[self.sig_start : self.sig_start +
				 gen_args['sig_len'] == gen_args['sig_len'])

	def test_sig_delay(self):
		assert self.d_len == gen_args['delay'] 

	def test_grad(self):
		assert(self.series[gen_args['sig_end'] + gen_args['delay']]
			- self.series[gen_args['sig_end' + gen_args['delay'] + 1]
			== gen_args['grad'])
