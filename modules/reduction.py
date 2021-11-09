from numpy.core.fromnumeric import cumsum
import tables
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def make_scree_plots(asml_file, times, clean_up=True, resolution=300):
    hdf5 = tables.open_file(asml_file, 'r')
    dir = os.path.dirname(asml_file)
    imgs = []
    for t in times:
        print('working on time: ' + str(t), end='\r')
        X = np.array(getattr(hdf5.root.particles, 'time_' + str(t)).read().tolist())
        X -= np.mean(X, axis=0)
        cov = (X.T @ X) / X.shape[0]
        evals, _ = np.linalg.eigh(cov)
        evals = evals[::-1] / evals.sum()
        print(evals)
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.scatter(list(range(len(evals))), np.cumsum(evals))
        plt.xlabel("principal component")
        plt.ylabel("cumulative variance explained")
        plt.title("time = {}".format(t))
        im_path = dir + '/scree_' + str(t) + '.png'
        plt.savefig(im_path)
        plt.close(fig)

        im = Image.open(im_path)
        rgb_im = Image.new('RGB', im.size, (255, 255, 255))  # white background
        rgb_im.paste(im, mask=im.split()[3])
        imgs.append(rgb_im)
        if clean_up:
            os.remove(im_path)
        if resolution is not None:
            imgs[0].save(dir + '/scree.pdf', "PDF", resolution = resolution, save_all = True, append_images = imgs[1:]) 



