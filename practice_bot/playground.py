import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

ends = pkl.load(open('ends_16.pkl', 'rb'))


prices = np.array([2 + np.sin(i / 120 * 2 * np.pi) for i in range(2120)])

test_prices = np.array([2 +np.sin(i / 120 * 2 * np.pi + 1) for i in range(1000)])


alpha = 0.3
gamma = 0.9

avf = np.ones([3, 3, 3, 3])

available =  [(0, 1, 2), (0, 2), (0, 1)]


long_pos = 0
short_pos = 0

state = [0, 0, 0]

uds=[]
for _ in range(500):
    state = [0, 0, 0]
    long_pos = 0
    short_pos = 0
    for t in range(2000):
        t += 119
        setstate(prices, t)
        if prices[t] == max(prices[t - 118: t + 1]) or prices[t] == min(prices[t - 118: t + 1]):
            newstate, action, reward = transition(state, avf, prices[t], 0.2)
            update = alpha * (reward + gamma * maxq(newstate) - avf[state[0], state[1], state[2], action])
            uds.append(update)
            avf[state[0], state[1], state[2], action] += update
            state = newstate[:]
            rewards.append(reward)

rewards = []
actions = []
states =[]
av = []
for t in range(len(test_prices) - 120):
    t += 119
    setstate(test_prices, t)
    if test_prices[t] == max(test_prices[t - 118: t + 1]) or test_prices[t] == min(test_prices[t - 118: t + 1]):
        newstate, action, reward = transition(state, avf, test_prices[t], 0.)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = newstate[:]
    else:
        actions.append(0)

plt.plot(test_prices)
for a, i in enumerate(actions):
    if i == 1:
        plt.axvline(a + 120, c='green')
    elif i == 2:
        plt.axvline(a + 120, c='red')

sum(rewards)
plt.show()


actions[:150]
av[:150]

def setstate(pr, t):
    if pr[t] == max(pr[t - 118: t + 1]):
        state[0] = 1
        state[2] += 1
    elif pr[t] == min(pr[t - 118: t + 1]):
        state[0] = 2
        state[2] += 1
    else:
        state[0] = 0
        state[2] = 0


def maxq(newstate):
    m = max([avf[newstate[0], newstate[1], newstate[2], i] for i in available[newstate[1]]])
    return m


def transition(state, avf, pr, ep):
    global long_pos
    global short_pos
    reward = 0
    action = np.random.choice(
                [0, 1], p = [1 - ep * (len(available[state[1]]) - 1),
                              ep * (len(available[state[1]]) - 1)])
    bestaction = available[state[1]][np.argmax([avf[state[0], state[1], state[2], i]
                            for i in available[state[1]]])]
    others = [i for i in available[state[1]] if i != bestaction]
    action = bestaction if action == 0 else np.random.choice(others)
    if state[1] == 1 and action == 2:
        reward = (1 - long_pos/pr)
        newstate = [state[0], 0, state[2]]
        long_pos = 0
    elif state[1] == 2 and action == 1:
        reward = (short_pos/pr - 1)
        newstate = [state[0], 0, state[2]]
        short_pos = 0
    elif state[1] == 0 and action == 1:
        long_pos = pr
        newstate = [state[0], 1, state[2]]
    elif state[1] == 0 and action == 2:
        short_pos = pr
        newstate = [state[0], 2, state[2]]
    else:
        newstate = state[:]
    return newstate, action, reward
