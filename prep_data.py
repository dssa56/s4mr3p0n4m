import pandas as pd
import numpy as np
import re
import pickle as pkl

dpm_ly = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

data_path = 'DAT_ASCII_EURUSD_M1_2015.csv'

df = pd.read_csv(data_path, sep=';')

df.columns = ['date_time', 'start', 'high', 'low', 'end', 'zero']

ends = np.array(df.end.tolist())


def convert_dt(dt):
    dt_li = dt.split()
    month_day = re.sub('2015', '', dt_li[0])
    mins = np.cumsum(dpm_ly)[int(month_day[:2]) - 1] * 1440
    mins += (int(month_day[2:]) - 1) * 1440
    mins += int(dt_li[1][:2]) * 60 + int(dt_li[1][2:4])
    return(mins)


def segment(times, x):
    ba = np.nonzero(np.array([times[i + 1] - times[i] > 720
                              for i in range(len(times) - 1)]))[0]
    ba = np.insert(ba, 0, 0)
    ba = np.append(ba, len(times) - 1)
    seg_times = np.array([times[ba[i] + 1: ba[i + 1] + 1]
                         for i in range(len(ba) - 1)])
    seg_x = np.array([x[ba[i] + 1: ba[i + 1] + 1] for i in range(len(ba) - 1)])
    return seg_times, seg_x


df.date_time = df.date_time.apply(convert_dt)
times = np.array(df.date_time.tolist())
pkl.dump(ends, open('ends_15.pkl', 'wb'))
pkl.dump(times, open('times_15.pkl', 'wb'))
