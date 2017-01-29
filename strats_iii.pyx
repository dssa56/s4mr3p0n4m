import numpy as np


def close_trades(price, sl, tp, trades, spread, pot):
    cdef int status = 0
    for i, trade in enumerate(trades):
        if trade[0] == 1:
            if price > trade[1] + tp:
                pot[0] -= trade[1] / price * trade[2] + spread
                trades.pop(i)
                if status == 0:
                    status = 1
            elif price < trade[1] - sl:
                pot[0] -= trade[1] / price * trade[2] + spread
                trades.pop(i)
                status = 2
        elif trade[0] == 2:
            if price < trade[1] - tp:
                pot[0] += trade[1] / price * trade[2] - spread
                trades.pop(i)
                if status == 0:
                    status = 1
            elif price > trade[1] + sl:
                pot[0] += trade[1] / price * trade[2] - spread
                trades.pop(i)
                status = 2
        return status


def open_trade(price, reg, trig, trades, bet, pot):
    if price > reg + trig:
        trades.append([2, price, pot[0] * bet])
        pot[0] -= pot[0] * bet
        return True
    elif price < reg - trig:
        trades.append([1, price, pot[0] * bet])
        pot[0] += pot[0] * bet
        return True
    return False


def strat_iii_record(params, double[:] prices, double[:] regs, int extra_time,
                     double spread):
    trades = []
    pot = np.array([1], np.float64)
    record = []
    money = []
    cdef:
        double trig = params[0]
        double sl = params[1]
        double tp = params[2]
        double bet = params[5]
        int i = 0
        double count_sw = 0, count_bw = 0
    sw = int(params[3])
    bw = int(params[4])
    while i < len(prices) - extra_time:
        b = close_trades(prices[i], sl, tp, trades, spread, pot)
        if b == 2:
            record.append((i, 3))
            money.append(pot[0])
            count_sw = sw
        elif b == 1:
            record.append((i, 3))
            money.append(pot[0])
        if count_sw == 0 and count_bw == 0:
            if open_trade(prices[i], regs[i], trig, trades, bet, pot):
                print(pot)
                record.append((i, 1))
                money.append(pot[0])
                count_bw = bw
        elif count_sw != 0:
            count_sw -= 1
        i += 1
        if count_bw != 0:
            count_bw -= 1
    while i < len(prices):
        if close_trades(prices[i], sl, tp, trades, spread, pot):
            money.append(pot[0])
            record.append((i, 3))
        i += 1
    return pot[0], record, money


def strat_iii_opt(params, double[:] prices, double[:] regs, int extra_time,
                  double spread):
    trades = []
    pot = np.array([1], np.float64)
    cdef:
        double trig = params[0]
        double sl = params[1]
        double tp = params[2]
        double bet = params[5]
        int i = 0
        double count_sw = 0, count_bw = 0
    sw = int(params[3])
    bw = int(params[4])
    while i < len(prices) - extra_time:
        if close_trades(prices[i], sl, tp, trades, spread, pot):
            count_sw = sw
        if count_sw == 0 and count_bw == 0:
            if open_trade(prices[i], regs[i], trig, trades, bet, pot):
                count_bw = bw
        elif count_sw != 0:
            count_sw -= 1
        i += 1
        if count_bw != 0:
            count_bw -= 1
    while i < len(prices):
        close_trades(prices[i], sl, tp, trades, spread, pot)
        i += 1
    if len(trades) > 0:
        return 10000
    return 1 / pot[0]
