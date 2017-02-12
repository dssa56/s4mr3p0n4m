import numpy as np


def get_mm_mi(datasets):
    return [max_move(ds, 5, 10) for ds in datasets]


def get_spike_data(datasets):
    si = [[minds[i + 1] for i in range(len(mmvs) - 1)
           if abs(mmvs[i + 1]) > 2 * abs(mmvs[i])]
          for minds, mmvs in datasets]

    ss = [[mmvs[i + 1] for i in range(len(mmvs) - 1)
           if abs(mmvs[i + 1]) > 2 * abs(mmvs[i])]
          for minds, mmvs in datasets]

    ps = [[mmvs[i] for i in range(len(mmvs) - 1)
          if abs(mmvs[i + 1]) > 2 * abs(mmvs[i])]
          for minds, mmvs in datasets]
    return zip(si, ss, ps)


def get_smalls(datasets, s_min, s_max):
    smalls = [get_spikes_by_size(ss, ps, si, s_min, s_max)
              for si, ss, ps in datasets]
    return smalls


def get_dist(datasets, smalls, win):
    dist = []
    dlist = []
    for ds, sm in zip(datasets, smalls):
        dist += lah_pr_dist(ds, sm, win)
        dlist.append(lah_pr_dist(ds, sm, win))
    return dist, dlist


def max_move(pr, move_len, n_moves):
    boxlen = move_len-1
    movelists = [[pr[i*boxlen + j * move_len] - pr[i*boxlen + (j+1) * move_len]
                  for j in range(n_moves)]
                 for i in range(len(pr) // (boxlen + 1))]
    maxlist = [ml[np.abs(ml).argmax()]
               for ml in movelists]
    maxinds = [move_len*np.abs(ml).argmax()+i*boxlen
               for i, ml in enumerate(movelists)]
    return maxinds, maxlist


def abs_spike_size_dist(pr, spks):
    up_dist = [pr[spk + 5] - pr[spk]
               for spk in spks if pr[spk + 5] - pr[spk] > 0]
    down_dist = [pr[spk + 5] - pr[spk]
                 for spk in spks if pr[spk + 5] - pr[spk] < 0]
    return up_dist, down_dist


def udlist(pr, spks):
    return [0 if pr[spk + 5] - pr[spk] < 0 else 1 for spk in spks]


def get_spikes_by_size(spks, prspks, sinds, size_min, size_max, abs_size_min):
    return [sinds[i] for i in range(len(spks))
            if abs(spks[i]/prspks[i]) > size_min
            and abs(spks[i]/prspks[i]) < size_max
            and abs(spks[i]) > abs_size_min]


def lah_dist(pr, s_inds, n_mins):
    aves = []
    for mi in range(n_mins):
        tot = 0
        for i in s_inds:
            if pr[i + 5] > pr[i]:
                if pr[i + 5] > pr[i + 5 + mi]:
                    tot += 1
            else:
                if pr[i + mi] > pr[i + 5 + mi]:
                    tot += 1
        tot /= len(s_inds)
        aves.append(tot)
    return aves


def lah_pr_dist(pr, s_inds, n_mins):
    prs = []
    for mi in range(n_mins):
        suprs = []
        for i in s_inds:
            size = abs(pr[i + 5] - pr[i])
            if pr[i + 5] > pr[i]:
                suprs.append((pr[i + 5] - pr[i + 5 + mi])/size)
            else:
                suprs.append((pr[i + 5 + mi] - pr[i + 5])/size)
        prs.append(suprs)
    return prs
