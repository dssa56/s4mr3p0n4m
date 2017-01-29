import pyximport
pyximport.install()
import canalyse as can
import numpy as np
import scipy.optimize as opt


def total_movement(double[:] prices):
    cdef int i
    cdef double dt, ave = 0
    for i in range(len(prices) - 1):
        dt = np.abs(prices[i] - prices[i + 1])
        ave += dt
    return ave


def average_movement(double[:] prices):
    cdef int i
    cdef double dt, ave = 0
    for i in range(len(prices) - 1):
        dt = np.abs(prices[i] - prices[i + 1])
        ave += dt
    return ave / len(prices)


def movement_entropy(double[:] prices):
    cdef int i
    cdef double dt, tot, entropy = 0
    tot = total_movement(prices)
    for i in range(len(prices) - 1):
        dt = np.abs(prices[i] - prices[i + 1]) / tot
        if dt != 0:
            entropy += dt * np.log2(dt)
    return - entropy


def ave_opt(x, *sample):
  s = sample[0][2 * (sample[1] - x): 2 * sample[1] + x]
  r, d = can.get_regs_devs(s, x, x)
  return np.sum(d)


def trendiness(double[:] prices, sense_len):
    assert len(prices) == 3 * sense_len
    reg_len = opt.brute(ave_opt, [slice(1, sense_len, 1)],
                        (prices, sense_len))
    r, _ = can.get_regs_devs(prices[2 * (sense_len - reg_len):],
                             reg_len, reg_len)
    return abs(sum(r) / sense_len - r[0])
