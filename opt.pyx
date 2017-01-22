from strats cimport sl_sg


cdef inline get_ind(int[7] lens, int[7] vals):
  cdef int [7] prods
  prods[6] = 1
  cdef int i
  for i in range(6):
    prods[6 - i - 1] = lens[i] * prods[6 - i]
  cdef sum = 0
  for i in range(7):
    sum += vals[i] * prods[i]
  return sum


def wrap_get_ind(int[7] lens, int[7] vals):
    return get_ind(lens, vals)

def opt_sl_sg(double[:] prices, md_dict, int[:] md_window, int[:] st_window,
               double[:] fact, double[:] a_sl, double[:] a_sg, double[:] b_sl,
               double[:] b_sg):
    cdef:
        int md_wl = len(md_window), st_wl = len(st_window)
        int fal = len(fact), a_sll = len(a_sl), a_sgl = len(a_sg)
        int b_sll = len(b_sl), b_sgl = len(b_sg)
        int md_w, st_w, fa, a_l, a_g, b_l, b_g
    cdef double [8, md_wl * st_wl * fal * a_sll * a_sgl * b_sll * b_sgl] result
    for md_w in range(md_wl):
      for st_w in range(st_wl):
        for fa in range(fal):
          for a_l in range(a_sll):
            for a_g in range(a_sgl):
              for b_l in range(b_sll):
                for b_g in range(b_sgl):
                  result[0, md_w * md_wl + st_w * st_wl + fa * fal +
                           a_l * a_sll + ]
