import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
print(module_dir)

import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import wasserstein as ws

num_experiments = 10
folders = ['seed_{}/dists'.format(seed) for seed in [3, 5, 7, 11, 13]]
obs_gaps = [0.01 + i*0.005 for i in range(num_experiments)]
avg_dist_folder = 'avg_dists'


fig  = plt.figure()
ax = fig.add_subplot(111)
t = np.array(list(range(1, 300+1)))



stb = []

for gap in obs_gaps:
    d = np.zeros(300)
    for folder in folders:
        for file in glob.glob('{}/prior_1_obs_gap_{}_*'.format(folder, gap)):
            df = pd.read_csv(file)
            d +=df['sinkhorn_div'].to_numpy()
    d /= len(folders)
    ax.semilogy((t*gap)[:15], d[:15], label='gap={:.4f}'.format(gap))
    stb.append(ws.find_stability_sma(d, window=20))
    data = {}
    data['sinkhorn_div'] = d 
    data['step'] = t
    pd.DataFrame(data).to_csv('{}/avg_dist_{}.csv'.format(avg_dist_folder, gap), index=False)

ax.set_ylabel('avg distance')
ax.set_xlabel('real time')
plt.legend()
plt.savefig('{}/avg_dist.png'.format(avg_dist_folder))
plt.close(fig)



fig_ = plt.figure()
ax_ = fig_.add_subplot(111)            
ax_.scatter(obs_gaps, np.array(stb))
ax_.set_ylabel('time to stabilize')
ax_.set_xlabel('obs gap')
plt.savefig('{}/avg_time_to_stabilize.png'.format(avg_dist_folder))


