import os
from skimage import io,transform
import numpy as np


PATH        = '../a/data/test_ruffles/'
SIZE = (151,151,3)
filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(PATH):
    if len(files)>1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
            io.imsave(files[i],transform.resize(io.imread(files[i]),SIZE))