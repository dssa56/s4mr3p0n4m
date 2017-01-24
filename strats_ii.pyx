cdef inline check_upper_dev_sell(price, mean, dev, dev_fact):
    if price > mean + dev * dev_fact:
        return True
    else:
        return False


cdef inline check_upper_dev_buy(price, mean, dev, dev_stoploss, dev_takeprofit):
    if price > mean + dev * dev_stoploss:
        return True
    if price < mean + dev * dev_takeprofit:
        return True
    else:
        return False


cdef inline check_upper_sr_sell(price, mean, dev, sr_fact):
    if price > mean + dev * sr_fact:
        return True
    else:
        return False


cdef inline check_upper_sr_buy(price, mean, dev, sr_stoploss, sr_takeprofit):
    if price > mean + dev * sr_stoploss:
        return True
    if price < mean + dev * sr_takeprofit:
        return True
    else:
        return False


cdef inline check_lower_dev_buy(price, mean, dev, dev_fact):
    if price < mean - dev * dev_fact:
        return True
    else:
        return False


cdef inline check_lower_dev_sell(price, mean, dev, dev_stoploss, dev_takeprofit):
    if price < mean - dev * dev_stoploss:
        return True
    if price > mean - dev * dev_takeprofit:
        return True
    else:
        return False


cdef inline check_lower_sr_buy(price, mean, dev, sr_fact):
    if price > mean - dev * sr_fact:
        return True
    else:
        return False


cdef inline check_lower_sr_sell(price, mean, dev, sr_stoploss, sr_takeprofit):
    if price < mean - dev * sr_stoploss:
        return True
    if price > mean - dev * sr_takeprofit:
        return True
    else:
        return False



def sr_dev_strat_ttd(double[:] params, double[:] prices, double[:] means,
                     double[:] devs):

    cdef:
        int i = 0, j = 0, status = 0 # 0: none, 1: u_s, 2: u_d, 3: u_s u_d, 4: l_s, 5: l_d, 6: l_s l_d, 7: u_s l_d, 8: l_s u_d
        double sr_rate, sr_mean, sr_dev, s_m, dev_rate, dev_mean, dev_dev, d_m
        double sr_bought, sr_sold, dev_bought, dev_sold
        double pot = 1
        double sr_stoploss = params[0]
        double sr_takeprofit = params[1]
        double dev_stoploss = params[2]
        double dev_takeprofit = params[3]
        double sr_fact = params[4]
        double dev_fact = params[5]
        double sr_bet = params[6]
        double dev_bet = params[7]

    record = []

    for i in range(prices):
        if status == 0:
            if check_lower_dev_buy(prices[i], means[i], devs[i], dev_fact):
                  bought = pot * dev_bet
                  status = 5
                  continue
            if check_upper_dev_sell(prices[i], means[i], devs[i], dev_fact):
                  sold = pot * dev_bet
                  status = 2
                  continue
            if check_lower_sr_buy(prices[i], means[i], devs[i], sr_fact):
                  bought = pot * sr_bet
                  status = 4
                  continue
            if check_upper_sr_sell(prices[i], means[i], devs[i], sr_fact):
                  bought = pot * sr_bet
                  status = 1
                  continue
        elif status == 1:
            if check_upper_sr_sell(prices[i], means[i], devs[i],
                                   sr_stoploss, sr_takeprofit):
                  pot = pot * dev_bet
                  status = 5
                  continue
            if check_upper_dev_sell(prices[i], means[i], devs[i], dev_fact):
                  sold = pot * dev_bet
                  status = 2
                  continue
            if check_lower_sr_buy(prices[i], means[i], devs[i], sr_fact):
                  bought = pot * sr_bet
                  status = 4
                  continue
            if check_upper_sr_sell(prices[i], means[i], devs[i], sr_fact):
                  bought = pot * sr_bet
                  status = 1
                  continue
