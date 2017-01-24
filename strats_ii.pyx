def sr_dev_strat_ttd(double[:] params, double[:] prices, double[:] means,
                 double[:] devs):

    cdef:
        int i = 0, j = 0
        double sr_rate, sr_mean, sr_dev, s_m, dev_rate, dev_mean, dev_dev, d_m
        double pot = 1
        double sr_stoploss = params[0]
        double sr_takeprofit = params[1]
        double dev_stoploss = params[2]
        double dev_takeprofit = params[3]
        double sr_fact = params[4]
        double dev_fact = params[5]
        double sr_bet = params[6]
        double dev_bet = params[7]

    while i < len(prices):

        if (prices[i] - means[i])**2 > (sr_fact * devs[i])**2:

            s_m = pot*sr_bet
            pot -= (prices[i]-means[i])/abs(prices[i]-means[i]) * s_m
            sr_rate = prices[i]
            sr_mean = means[i]
            sr_dev = devs[i]

            while (((prices[i] - sr_mean)**2 < (sr_stoploss * sr_dev)**2)
                   and (prices[i] - sr_mean)**2 >
                   (sr_takeprofit * sr_dev)**2):

                  if j == 0:
                      if ((prices[i] - means[i])**2 >
                          (dev_fact * devs[i])**2):

                          d_m = pot*dev_bet
                          pot -= ((prices[i] - means[i]) /
                                  abs(prices[i] - means[i]) * d_m)
                          dev_rate = prices[i]
                          dev_mean = means[i]
                          dev_dev = devs[i]
                          j = 1

                  elif j == 1:
                      if (((prices[i] - dev_mean)**2 >
                          (dev_stoploss * dev_dev)**2)
                          or ((prices[i] - dev_mean)**2 <
                          (dev_takeprofit * dev_dev)**2)):

                          j = 0
                          pot += ((prices[i] - means[i]) /
                                  abs(prices[i] - means[i]) * d_m
                                  * dev_rate / prices[i])

                  i += 1
            pot += ((prices[i] - means[i]) / abs(prices[i] - means[i])
                    * s_m * sr_rate / prices[i])


        if j == 1:
            if (((prices[i] - dev_mean)**2 > (dev_stoploss * dev_dev)**2) or
                ((prices[i] - dev_mean)**2 <
                (dev_takeprofit * dev_dev)**2)):

                j = 0
                pot += ((prices[i] - means[i]) /
                        abs(prices[i] - means[i]) * d_m
                        * sr_rate / prices[i])

        elif j == 0:
            if (prices[i] - means[i])**2 > (dev_fact * devs[i])**2:
                d_m = pot*dev_bet
                pot -= ((prices[i] - means[i]) /
                        abs(prices[i] - means[i]) * d_m)
                dev_rate = prices[i]
                dev_mean = means[i]
                dev_dev = devs[i]
                j = 1

        i += 1
    return pot
