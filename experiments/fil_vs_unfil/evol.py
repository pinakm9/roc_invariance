import os, sys 
from pathlib import Path
from cv2 import repeat
from matplotlib import projections 

script_dir = Path(os.path.abspath(''))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 
import numpy as np 
import tables
import utility as ut
import Lorenz63_xz as lorenz
import wasserstein as ws
import pandas as pd

obs_gap = 0.01
n_particles = 500
ev_time = 500
# load the attractor
attractor = np.genfromtxt('../../models/attractor_10000.csv', delimiter=',')[:2000]
# gather data for filtering distribution
asml_file_1 = '../L63_1_vs_3_seed_3_const_obs_cov/data/prior_1_obs_gap_0.01/assimilation.h5'
asml_file_2 = '../L63_1_vs_3_seed_3_const_obs_cov/data/prior_4_obs_gap_0.01/assimilation.h5'
hdf_1 = tables.open_file(asml_file_1, 'r')
hdf_2 = tables.open_file(asml_file_2, 'r')

def get_ensembles(t):
    ens_1 = np.array(getattr(hdf_1.root.particles, 'time_{}'.format(t)).read().tolist())
    ens_2 = np.array(getattr(hdf_2.root.particles, 'time_{}'.format(t)).read().tolist())
    return ens_1, ens_2


# execute once
"""
e1, e2 = np.zeros((n_particles, ev_time, 3)), np.zeros((n_particles, ev_time, 3))
e1[:, 0, :], e2[:, 0, :] = get_ensembles(0)
_, gen_path = lorenz.get_model(x0=np.zeros(3), shift=0., obs_gap=obs_gap, ev_time=ev_time, obs_cov=0.1)
for n in range(n_particles):
    e1[n, :, :] = gen_path(e1[n, 0, :], length=ev_time)
    e2[n, :, :] = gen_path(e2[n, 0, :], length=ev_time)
np.save('data/unfiltered_1.npy', e1)
np.save('data/unfiltered_2.npy', e2)
#"""
uf_1 = np.load('data/unfiltered_1.npy')
uf_2 = np.load('data/unfiltered_2.npy')

# compute distance
"""
wd = np.zeros(ev_time)
for i in range(ev_time):
    wd[i] = ws.sinkhorn_div_tf(uf_1[:, i, :], uf_2[:, i, :])**0.5
    gap = '{:.2f}'.format(obs_gap).replace('.', '_')
pd.DataFrame(wd).to_csv('data/unfiltered_1_vs_3_obs_gap_{}.csv'.format(obs_gap), header=None, index=None)
#"""
dist_1 = pd.read_csv('data/prior_1_obs_gap_0.01_vs_prior_4_obs_gap_0.01.csv', delimiter=',')['sinkhorn_div'].to_numpy()
dist_2 = np.genfromtxt('data/unfiltered_prior_1_obs_gap_0.01_vs_prior_4_obs_gap_0.01.csv', delimiter=',')


fig = plt.figure(figsize=(20, 10))
ax_f = fig.add_subplot(121, projection='3d')
ax_u = fig.add_subplot(122, projection='3d')


def animator(t):
    ax_f.clear()
    ens_1, ens_2 = get_ensembles(t)
    ax_f.scatter(attractor[:, 0], attractor[:, 1], attractor[:, 2], c='lightblue')
    ax_f.scatter(ens_1[:, 0], ens_1[:, 1], ens_1[:, 2], c='deeppink', label='good prior')
    ax_f.scatter(ens_2[:, 0], ens_2[:, 1], ens_2[:, 2], c='orange', label='bad prior')
    ax_f.set_title('filtered, time = {:.2f}, dist = {:.2f}'.format(obs_gap * t, dist_1[t]))


    ax_u.clear()
    ens_1, ens_2 = uf_1[:, t, :], uf_2[:, t, :]
    ax_u.scatter(attractor[:, 0], attractor[:, 1], attractor[:, 2], c='lightblue')
    ax_u.scatter(ens_1[:, 0], ens_1[:, 1], ens_1[:, 2], c='deeppink', label='good prior')
    ax_u.scatter(ens_2[:, 0], ens_2[:, 1], ens_2[:, 2], c='orange', label='bad prior')
    ax_u.set_title('unfiltered, time = {:.2f}, dist = {:.2f}'.format(obs_gap * t, dist_2[t]))

    ax_f.set_xlim(-20, 20)
    ax_f.set_ylim(-30, 30)
    ax_f.set_zlim(0, 50)
    ax_u.set_xlim(-20, 20)
    ax_u.set_ylim(-30, 30)
    ax_u.set_zlim(0, 50)
anim = FuncAnimation(fig=fig, func=animator, frames=ev_time, repeat=False)
anim.save('data/filtered_vs_unfiltered.mp4', writer='ffmpeg', fps=12)
#plt.show()





