from strats cimport get_diffs, stop_prs
import numpy as np


cdef inline get_ind(int[:] lens, int[:] vals):
  cdef int [7] prods
  prods[6] = 1
  cdef int i
  for i in range(6):
    prods[6 - i - 1] = lens[6 - i] * prods[6 - i]
  cdef sum = 0
  for i in range(7):
    sum += vals[i] * prods[i]
  return sum


def wrap_get_ind(int[:] lens, int[:] vals):
    return get_ind(lens, vals)


def get_best(x):
    cdef int n = len(x[:, 0])
    best = np.zeros([n], np.float64)
    cdef int i
    for i in range(len(x[0])):
        if x[n - 1, i] > best[n - 1]:
            best = x[:, i]
    return best


def opt_sl_sg(double[:] prices, md_dict, int[:] md_window, int[:] st_window,
               double[:] fact, double[:] a_sl, double[:] a_sg, double[:] b_sl,
               double[:] b_sg, stop_params):
    cdef:
        int md_wl = len(md_window), st_wl = len(st_window)
        int fal = len(fact), a_sll = len(a_sl), a_sgl = len(a_sg)
        int b_sll = len(b_sl), b_sgl = len(b_sg)
        int md_w, st_w, fa, a_l, a_g, b_l, b_g, ind
        int lens_l [7]
        int vals_l [7]
    result = np.zeros([8, md_wl*st_wl*fal*a_sll*a_sgl*b_sll*b_sgl])
    for md_w in range(md_wl):
      for st_w in range(st_wl):
        for fa in range(fal):
          for a_l in range(a_sll):
            for a_g in range(a_sgl):
              for b_l in range(b_sll):
                for b_g in range(b_sgl):
                  lens_l[0] = md_wl
                  lens_l[1] = st_wl
                  lens_l[2] = fal
                  lens_l[3] = a_sll
                  lens_l[4] = a_sgl
                  lens_l[5] = b_sll
                  lens_l[6] = b_sgl
                  vals_l[0] = md_w
                  vals_l[1] = st_w
                  vals_l[2] = fa
                  vals_l[3] = a_l
                  vals_l[4] = a_g
                  vals_l[5] = b_l
                  vals_l[6] = b_g
                  ind = get_ind(lens_l,
                                vals_l)
                  result[0, ind] = md_window[md_w]
                  result[1, ind] = st_window[st_w]
                  result[2, ind] = fact[fa]
                  result[3, ind] = a_sl[a_l]
                  result[4, ind] = a_sg[a_g]
                  result[5, ind] = b_sl[b_l]
                  result[6, ind] = b_sg[b_g]
                  diffs = get_diffs(prices, md_dict, md_window[md_w],
                                          st_window[st_w], fact[fa],
                                          a_sl[a_l], b_sg[a_g],
                                          b_sl[b_l], b_sg[b_g])
                  result[7, ind] = stop_prs(stop_params, diffs)

    return get_best(result)
