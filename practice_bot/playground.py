import numpy as np
import matplotlib.pyplot as plt



prices = np.array([2 + np.sin(i / 120 * 2 * np.pi) for i in range(2120)])

test_prices = np.array([2 +np.cos(i / 120 * 2 * np.pi + 1) for i in range(1000)])

alpha = 0.2
gamma = 0.9
ep = 0.1

avf = np.zeros([3, 3, 3])

available =  [(0, 1, 2), (0, 2), (0, 1)]

state = [0, 0]

long_pos = 0
short_pos = 0

for _ in range(500):
    for t in range(2000):
        t += 119
        if prices[t] == max(prices[t - 118: t + 1]):
            state[0] = 1
        elif prices[t] == min(prices[t - 118: t + 1]):
            state[0] = 2
        else:
            state[0] = 0
        newstate, action, reward = transition(state, avf, t, 0.1)
        avf[state[0], state[1], action] += alpha * (reward + gamma * maxq(newstate) - avf[state[0], state[1], action])
        state = newstate

rewards = []
actions = []
log = []
for t in range(len(test_prices) - 120):
    t += 120
    if test_prices[t] == max(test_prices[t - 118: t + 1]):
        state[0] = 1
    elif test_prices[t] == min(test_prices[t - 118: t + 1]):
        state[0] = 2
    else:
        state[0] = 0
    newstate, action, reward = transition(state, avf, t, 0)
    print((state, newstate), reward, action)
    log.append([(state, newstate), reward, action])
    rewards.append(reward)
    actions.append(action)
    state = newstate

log

plt.plot(test_prices)
for a, i in enumerate(actions):
    if i == 1:
        plt.axvline(a + 120, c='green')
    elif i == 2:
        plt.axvline(a + 120, c='red')

sum(rewards)

log

plt.show()


def maxq(newstate):
    m = max([avf[newstate[0], newstate[1], i] for i in available[newstate[1]]])
    return m


def transition(state, avf, t, ep):
    global long_pos
    global short_pos
    reward = 0
    action = np.random.choice(
                [0, 1], p = [1 - ep * (len(available[state[1]]) - 1),
                              ep * (len(available[state[1]]) - 1)])
    bestaction = available[state[1]][np.argmax([avf[state[0], state[1], i]
                            for i in available[state[1]]])]
    others = [i for i in available[state[1]] if i != bestaction]
    action = bestaction if action == 0 else np.random.choice(others)
    if state[1] == 1 and action == 2:
        reward = (1 - long_pos/prices[t])
        newstate = [state[0], 0]
    elif state[1] == 2 and action == 1:
        reward = (short_pos/prices[t] - 1)
        newstate = [state[0], 0]
    elif state[1] == 0 and action == 1:
        long_pos = prices[t]
        newstate = [state[0], 1]
    elif state[1] == 0 and action == 2:
        short_pos = prices[t]
        newstate = [state[0], 2]
    else:
        newstate = state
    return newstate, action, reward
