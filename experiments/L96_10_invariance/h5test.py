import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import tables
import wasserstein as ws

folders = ['data/65b8345b62e2c11a3f6f4396a05dfec9_prior_1', 'data/dfbb09f7c507436b3ed387cf71d784c4_prior_3']
"""
hdf5 = tables.open_file(folder + '/assimilation.h5', mode='r')
hd5 = tables.open_file(folder + '/assimilation.h5', mode='r')
print(hdf5.root.particles.time_259.read().tolist())
print(len(hd5.root.observation.read().tolist()))
hdf5.close()
hd5.close()
"""
bd = ws.BatchDist(folders, folders, 'dists')
bd.run(gap=1, ev_time=None, plot=True)