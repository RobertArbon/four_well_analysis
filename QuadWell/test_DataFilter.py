import numpy as np
from discretization_optimization import DataFilter
from glob import glob
traj_paths = glob('data/*.npy')

X = [np.load(traj_path) for traj_path in traj_paths]
old_lengths = np.array([len(x) for x in X])


old_tot_length = np.sum(old_lengths)

df = DataFilter(fraction=0.2)
df.fit(X)
X_new = df.predict(X)


new_lengths = np.array([len(x) for x in X_new])
new_tot_length = np.sum(new_lengths)

print('Old lengths = {}'.format(old_lengths))
print('New lengths = {}'.format(new_lengths))
print('New {0} Old  {1} Fraction {2}'.format(old_tot_length, new_tot_length, new_tot_length/old_tot_length))