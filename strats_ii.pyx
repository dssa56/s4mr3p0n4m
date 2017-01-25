import numpy as np

cdef inline check_upper_dev_sell(price, mean, dev, dev_fact, dev_stoploss):
    if price > mean + dev_stoploss:
        return False
    if price > mean + dev_fact:
        return True
    else:
        return False


cdef inline check_upper_dev_buy(price, mean, dev, dev_stoploss,
                                dev_takeprofit, time, dev_tmax):
    if time > dev_tmax:
        return True
    if price > mean + dev_stoploss:
        return True
    if price < mean + dev_takeprofit:
        return True
    else:
        return False


cdef inline check_upper_sr_sell(price, mean, dev, sr_fact, sr_stoploss):
    if price > mean + sr_stoploss:
        return False
    if price > mean + sr_fact:
        return True
    else:
        return False


cdef inline check_upper_sr_buy(price, mean, dev, sr_stoploss, sr_takeprofit,
                                time, sr_tmax):
    if time > sr_tmax:
        return True
    if price > mean + sr_stoploss:
        return True
    if price < mean + sr_takeprofit:
        return True
    else:
        return False


cdef inline check_lower_dev_buy(price, mean, dev, dev_fact, dev_stoploss):
    if price < mean - dev_stoploss:
        return False
    if price < mean - dev_fact:
        return True
    else:
        return False


cdef inline check_lower_dev_sell(price, mean, dev, dev_stoploss,
                                 dev_takeprofit, time, dev_tmax):
    if time > dev_tmax:
        return True
    if price < mean - dev_stoploss:
        return True
    if price > mean - dev_takeprofit:
        return True
    else:
        return False


cdef inline check_lower_sr_buy(price, mean, dev, sr_fact, sr_stoploss):
    if price < mean - sr_stoploss:
      return False
    if price < mean - sr_fact:
        return True
    else:
        return False


cdef inline check_lower_sr_sell(price, mean, dev, sr_stoploss, sr_takeprofit,
                                time, sr_tmax):
    if time > sr_tmax:
        return True
    if price < mean - sr_stoploss:
        return True
    if price > mean - sr_takeprofit:
        return True
    else:
        return False


cdef inline buy_lower_dev(dev_store, pot, price, mean, dev, dev_bet, i):
    np.copyto(dev_store, [price, mean, dev, pot * dev_bet, i])
    return pot * (1 + dev_bet)


cdef inline buy_lower_sr(sr_store, pot, price, mean, dev, sr_bet, i):
    np.copyto(sr_store, [price, mean, dev, pot * sr_bet, i])
    return pot * (1 + sr_bet)


cdef inline sell_lower_dev(dev_store, price, pot):
    return pot - dev_store[0] / price * dev_store[3]


cdef inline sell_lower_sr(sr_store, price, pot):
    return pot - sr_store[0] / price * sr_store[3]


cdef inline sell_upper_dev(dev_store, pot, price, mean, dev, dev_bet, i):
    np.copyto(dev_store, [price, mean, dev, pot * dev_bet, i])
    return pot * (1 - dev_bet)


cdef inline sell_upper_sr(sr_store, pot, price, mean, dev, sr_bet, i):
    np.copyto(sr_store, [price, mean, dev, pot * sr_bet, i])
    return pot * (1 - sr_bet)


cdef inline buy_upper_dev(dev_store, price, pot):
    return pot + dev_store[0] / price * dev_store[3]


cdef inline buy_upper_sr(sr_store, price, pot):
    return pot + sr_store[0] / price * sr_store[3]


def sr_dev_strat_ttd(params, double[:] prices, double[:] means,
                     double[:] devs):

    cdef:
        int i, status = 0 # 0: none, 1: u_s, 2: u_d, 3: u_s u_d, 4: l_s, 5: l_d, 6: l_s l_d
        double pot = 1
        double sr_stoploss = params[0]
        double sr_takeprofit = params[1]
        double dev_stoploss = params[2]
        double dev_takeprofit = params[3]
        double sr_fact = params[4]
        double dev_fact = params[5]
        double sr_bet = params[6]
        double dev_bet = params[7]
        int sr_tmax = params[8]
        int dev_tmax = params[9]

    sr_store = np.zeros([5], np.float64) # 0: rate, 1: mean, 2: dev, 3: bought/sold 4: time
    dev_store = np.zeros([5], np.float64)
    record = []

    for i in range(len(prices)):
        if status == 0:
            if check_lower_dev_buy(prices[i], means[i], devs[i],
                                   dev_fact, dev_stoploss):
                pot = buy_lower_dev(dev_store, pot, prices[i],
                                    means[i], devs[i], dev_bet, i)
                record.append((i, 0, pot)) # 0: od, 1: os, 2: cd, 3: cs
                status = 5
                continue
            if check_upper_dev_sell(prices[i], means[i], devs[i],
                                    dev_fact, dev_stoploss):
                pot = sell_upper_dev(dev_store, pot, prices[i],
                                    means[i], devs[i], dev_bet, i)
                record.append((i, 0, pot))
                status = 2
                continue
            if check_lower_sr_buy(prices[i], means[i], devs[i],
                                  sr_fact, sr_stoploss):
                pot = buy_lower_sr(sr_store, pot, prices[i], means[i],
                                   devs[i], sr_bet, i)
                record.append((i, 1, pot))
                status = 4
                continue
            if check_upper_sr_sell(prices[i], means[i], devs[i],
                                   sr_fact, sr_stoploss):
                pot = sell_upper_sr(sr_store, pot, prices[i], means[i],
                                    devs[i], sr_bet, i)
                record.append((i, 1, pot))
                status = 1
                continue
        if status == 1:
            if check_upper_sr_buy(prices[i], sr_store[1], sr_store[2],
                                  sr_stoploss, sr_takeprofit, i - sr_store[4],
                                  sr_tmax):
                pot = buy_upper_sr(sr_store, prices[i], pot)
                record.append((i, 3, pot))
                status = 0
                continue
            if check_upper_dev_sell(prices[i], means[i], devs[i],
                                    dev_fact, dev_stoploss):
                pot = sell_upper_dev(dev_store, pot, prices[i],
                                    means[i], devs[i], dev_bet, i)
                record.append((i, 0, pot))
                status = 3
                continue
        if status == 2:
            if check_upper_dev_buy(prices[i], dev_store[1], dev_store[2],
                             dev_stoploss, dev_takeprofit, i - dev_store[4],
                             dev_tmax):
                pot = buy_upper_dev(dev_store, prices[i], pot)
                record.append((i, 2, pot))
                status = 0
                continue
            if check_upper_sr_sell(prices[i], means[i], devs[i],
                                   sr_fact, sr_stoploss):
                pot = sell_upper_sr(sr_store, pot, prices[i], means[i],
                                    devs[i], sr_bet, i)
                record.append((i, 1, pot))
                status = 3
                continue
        if status == 3:
            if check_upper_dev_buy(prices[i], dev_store[1], dev_store[2],
                             dev_stoploss, dev_takeprofit, i - dev_store[4],
                             dev_tmax):
                pot = buy_upper_dev(dev_store, prices[i], pot)
                record.append((i, 2, pot))
                status = 1
            if check_upper_sr_buy(prices[i], sr_store[1], sr_store[2],
                                  sr_stoploss, sr_takeprofit, i - sr_store[4],
                                  sr_tmax):
                pot = buy_upper_sr(sr_store, prices[i], pot)
                record.append((i, 3, pot))
                if status == 1:
                    status = 0
                else:
                    status = 2
                continue
        if status == 4:
            if check_lower_sr_sell(prices[i], sr_store[1], sr_store[2],
                                  sr_stoploss, sr_takeprofit, i - sr_store[4],
                                  sr_tmax):
                pot = sell_lower_sr(sr_store, prices[i], pot)
                record.append((i, 3, pot))
                status = 0
                continue
            if check_lower_dev_buy(prices[i], means[i], devs[i],
                                    dev_fact, dev_stoploss):
                pot = buy_lower_dev(dev_store, pot, prices[i],
                                    means[i], devs[i], dev_bet, i)
                record.append((i, 0, pot))
                status = 6
                continue
        if status == 5:
            if check_lower_dev_sell(prices[i], dev_store[1], dev_store[2],
                             dev_stoploss, dev_takeprofit, i - dev_store[4],
                             dev_tmax):
                pot = sell_lower_dev(dev_store, prices[i], pot)
                record.append((i, 2, pot))
                status = 0
                continue
            if check_lower_sr_buy(prices[i], means[i], devs[i],
                                   sr_fact, sr_stoploss):
                pot = buy_lower_sr(sr_store, pot, prices[i], means[i],
                                    devs[i], sr_bet, i)
                record.append((i, 1, pot))
                status = 6
                continue
        if status == 6:
            if check_lower_dev_sell(prices[i], dev_store[1], dev_store[2],
                             dev_stoploss, dev_takeprofit, i - dev_store[4],
                             dev_tmax):
                pot = sell_lower_dev(dev_store, prices[i], pot)
                record.append((i, 2, pot))
                status = 4
            if check_lower_sr_sell(prices[i], sr_store[1], sr_store[2],
                                  sr_stoploss, sr_takeprofit, i - sr_store[4],
                                  sr_tmax):
                pot = sell_lower_sr(sr_store, prices[i], pot)
                record.append((i, 3, pot))
                if status == 4:
                    status = 0
                else:
                    status = 5
                continue



    if status == 0:
        return pot, record
    if status == 1:
        pot = buy_upper_sr(sr_store, prices[-1], pot)
        return pot, record
    if status == 2:
        pot = buy_upper_dev(dev_store, prices[-1], pot)
        return pot, record
    if status == 3:
        pot = buy_upper_dev(dev_store, prices[-1], pot)
        pot = buy_upper_sr(sr_store, prices[-1], pot)
        return pot, record
    if status == 4:
        pot = sell_lower_sr(sr_store, prices[-1], pot)
        return pot, record
    if status == 5:
        pot = sell_lower_dev(dev_store, prices[-1], pot)
        return pot, record
    if status == 6:
        pot = sell_lower_dev(dev_store, prices[-1], pot)
        pot = sell_lower_sr(sr_store, prices[-1], pot)
        return pot, record
