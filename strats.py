def strat_A(lah, init, perc):
    to_trade = perc*init
    money = []
    for l in lah:
        init += l*to_trade
        to_trade = perc*init
        money.append(init)
    return money
