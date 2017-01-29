p_train = pkl.load(open('ends_16.pkl', 'rb'))
d_train = pkl.load(open('md_dict_16.pkl', 'rb'))
times = pkl.load(open('times_16.pkl', 'rb'))

p_test = pkl.load(open('ends_15.pkl', 'rb'))
d_test = pkl.load(open('md_dict_15.pkl', 'rb'))

pd = {'md_window': np.array([50]).astype(np.int32),
      'st_window': np.array([1, 2, 3, 4, 5]).astype(np.int32),
      'fact': np.array([1]).astype(np.float64),
      'a_sl': np.array([5.5, 6, 6.5, 7]).astype(np.float64),
      'a_sg': np.array([-0.5, 0, 0.5]).astype(np.float64),
      'b_sl': np.array([5.5, 6, 6.5, 7]).astype(np.float64),
      'b_sg': np.array([-0.5, 0, 1]).astype(np.float64)}

opt.opt_sl_sg(p_train, d_train, pd['md_window'],
                pd['st_window'], pd['fact'], pd['a_sl'], pd['a_sg'], pd['b_sl'],
                pd['b_sg'], [10, 10, 7])

diffs = strats.wrap_get_diffs(p_train, d_train, 50, 4, 1,
                              6.5, 0, 6, 1)

sopt.brute(strats.opt_stop_prs, (slice(5, 20, 1), slice(5, 20, 1), slice(5, 20, 1)),
            args = (diffs,))

y = [8, 5, 5]

r = strats.strat(y, diffs)

plt.plot(r)
plt.show()
