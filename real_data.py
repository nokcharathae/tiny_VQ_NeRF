import os

os.environ["KERAS_BACKEND"] = "tensorflow"

# Setting random seed to obtain reproducible results.
import tensorflow as tf

tf.random.set_seed(42)

import keras
from keras import layers

import os
import glob
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


# Initialize global variables.
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 1
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 200
Embbed_size = 64
save_interval =50

# Download the data if it does not already exist.
url = (
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
)
data = keras.utils.get_file(origin="tiny_nerf_data.npz")

data = np.load(data)
images = data["images"]
im_shape = images.shape
print(im_shape)
(num_images, H, W, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])

print(focal)