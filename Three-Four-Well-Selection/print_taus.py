import numpy as np

def get_taus(ts):
    grouped_ts = []
    i = 0
    while i < len(ts) - 1:
        if np.abs(ts[i + 1] - ts[i]) < 2:
            grouped_ts.append(int(np.mean(ts[i:i + 2])))
            i += 2
        else:
            grouped_ts.append(int(ts[i]))
            i += 1
    taus = [int((grouped_ts[i]+grouped_ts[i+1])/2) for i in range(len(grouped_ts)-1)]
    return taus


ts = np.load('timescales.npy')
taus = get_taus(ts)
print(taus)