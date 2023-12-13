import glob
import imageio.v2 as imageio
from tqdm import tqdm

import keras

model = keras.models.load_model('nerf_model_200_64_64.h5')

def create_gif(path_to_images, name_gif):
    filenames = glob.glob(path_to_images)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    kargs = {"loop":0, "duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)


create_gif("images3/*.JPG", "NeRF.gif")