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
import json
import pandas as pd
from plyfile import PlyData, PlyElement


# Initialize global variables.
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 1
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 200
Embbed_size = 512
save_interval =50

import open3d as o3d

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
print(poses.shape)

# Plot a random image from the dataset for visualization.
# plt.imshow(images[np.random.randint(low=0, high=num_images)])
# plt.show()


def encode_position(x):
    """Encodes the position into its corresponding Fourier feature.

    Args:
        x: The input coordinate.

    Returns:
        Fourier features tensors of the position.
    """
    positions = [x]
    for i in range(POS_ENCODE_DIMS):
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0**i * x))
    return tf.concat(positions, axis=-1)


def get_rays(height, width, focal, pose):
    """Computes origin point and direction vector of rays.

    Args:
        height: Height of the image.
        width: Width of the image.
        focal: The focal length between the images and the camera.
        pose: The pose matrix of the camera.

    Returns:
        Tuple of origin point and direction vector for rays.
    """
    # Build a meshgrid for the rays.
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )

    # Normalize the x axis coordinates.
    transformed_i = (i - width * 0.5) / focal

    # Normalize the y axis coordinates.
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)

    # Get the camera matrix.
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))

    # Return the origins and directions.
    return (ray_origins, ray_directions)


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    """Renders the rays and flattens it.

    Args:
        ray_origins: The origin points for rays.
        ray_directions: The direction unit vectors for the rays.
        near: The near bound of the volumetric scene.
        far: The far bound of the volumetric scene.
        num_samples: Number of sample points in a ray.
        rand: Choice for randomising the sampling strategy.

    Returns:
       Tuple of flattened rays and sample points on each rays.
    """
    # Compute 3D query points.
    # Equation: r(t) = o+td -> Building the "t" here.
    t_vals = tf.linspace(near, far, num_samples)
    if rand:
        # Inject uniform noise into sample space to make the sampling
        # continuous.
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    # Equation: r(t) = o + td -> Building the "r" here.
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)


def map_fn(pose):
    """Maps individual pose to flattened rays and sample points.

    Args:
        pose: The pose matrix of the camera.

    Returns:
        Tuple of flattened rays and sample points corresponding to the
        camera pose.
    """
    (ray_origins, ray_directions) = get_rays(height=H, width=W, focal=focal, pose=pose)
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        num_samples=NUM_SAMPLES,
        rand=True,
    )
    return (rays_flat, t_vals)


print("initial")
# Create the training split.
split_index = int(num_images * 0.8)

# Split the images into training and validation.
train_images = images[:split_index]
val_images = images[split_index:]

# Split the poses into training and validation.
train_poses = poses[:split_index]
val_poses = poses[split_index:]

# Make the training pipeline.
train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
train_ray_ds = train_pose_ds.map(map_fn, num_parallel_calls=AUTO)
training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
train_ds = (
    training_ds.shuffle(BATCH_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

# Make the validation pipeline.
val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
val_ray_ds = val_pose_ds.map(map_fn, num_parallel_calls=AUTO)
validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds))
val_ds = (
    validation_ds.shuffle(BATCH_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

print("set_VQ")

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

print("set_model")


def get_nerf_model(num_layers, num_pos, latent_dim=32, num_embeddings=Embbed_size):
    """Generates the NeRF neural network.

    Args:
        num_layers: The number of MLP layers.
        num_pos: The number of dimensions of positional encoding.

    Returns:
        The `keras` model.
    """
    inputs = keras.Input(shape=(num_pos, 2 * 3 * POS_ENCODE_DIMS + 3))
    x = inputs
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")

    for i in range(num_layers):
        x = layers.Dense(units=64, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            # Inject residual connection.
            x = layers.concatenate([x, inputs], axis=-1)
    quantized_latents = vq_layer(x)
    outputs = layers.Dense(units=4)(quantized_latents)
    return keras.Model(inputs=inputs, outputs=outputs)


def render_rgb_depth(model, rays_flat, t_vals, rand=True, train=True):
    """Generates the RGB image and depth map from model prediction.

    Args:
        model: The MLP model that is trained to predict the rgb and
            volume density of the volumetric scene.
        rays_flat: The flattened rays that serve as the input to
            the NeRF model.
        t_vals: The sample points for the rays.
        rand: Choice to randomise the sampling strategy.
        train: Whether the model is in the training or testing phase.

    Returns:
        Tuple of rgb image and depth map.
    """
    # Get the predictions from the nerf model and reshape it.
    if train:
        predictions = model(rays_flat)
    else:
        predictions = model.predict(rays_flat)
    predictions = tf.reshape(predictions, shape=(BATCH_SIZE, H, W, NUM_SAMPLES, 4))

    # Slice the predictions into rgb and sigma.
    rgb = tf.sigmoid(predictions[..., :-1])
    sigma_a = tf.nn.relu(predictions[..., -1])

    # Get the distance of adjacent intervals.
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    # delta shape = (num_samples)
    if rand:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, H, W, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta)
    else:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])

    # Get transmittance.
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = tf.math.cumprod(exp_term + epsilon, axis=-1, exclusive=True)
    weights = alpha * transmittance
    rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

    if rand:
        depth_map = tf.reduce_sum(weights * t_vals, axis=-1)
    else:
        depth_map = tf.reduce_sum(weights * t_vals[:, None, None], axis=-1)
    return (rgb, depth_map)

print("training model")


def depth_to_point(rgb, depth, rays_o, rays_d):
    # 입력으로 받은 텐서들을 numpy로 변환 후 reshape
    rgb_cpu = np.reshape(rgb.numpy(), [-1, 3])  # shape: [10000, 3]
    depth_cpu = np.reshape(depth.numpy(), [-1])  # shape: [10000]
    rays_o_np = np.reshape(rays_o.numpy(), [-1, 3])  # shape: [10000, 3]
    rays_d_np = np.reshape(rays_d.numpy(), [-1, 3])  # shape: [10000, 3]

    # depth에서 nan이 아닌 값들의 index를 찾음
    idx = np.where(np.logical_not(np.isnan(depth_cpu)))

    # valid_coord 계산
    valid_coord = rays_o_np[idx] + rays_d_np[idx] * depth_cpu[idx][:, np.newaxis]
    
    # valid_pt 계산
    valid_pt = np.concatenate([valid_coord, rgb_cpu[idx]], axis=-1)

    return valid_pt



'''
def depth_to_point(rgb, depth, rays_o, rays_d):
    # shape = tf.cast(tf.sqrt(tf.cast(tf.shape(rays_o)[0], tf.float32)), tf.int32)
    # print(shape)
    # rays_o_np = tf.reshape(rays_o, [shape, shape, 3])
    # rays_d_np = tf.reshape(rays_d, [shape, shape, 3])


    # Get indices where depth is not NaN
    idx = tf.where(tf.math.logical_not(tf.math.is_nan(depth)))

    # Convert 2D indices to 3D indices
    idx_3d = tf.stack([idx[:, 0], idx[:, 1], tf.zeros_like(idx[:, 0])], axis=-1)

    # Calculate valid coordinates
    valid_coord = tf.gather_nd(rays_o, idx_3d) + tf.gather_nd(rays_d, idx_3d) * tf.expand_dims(tf.gather_nd(depth, idx), axis=-1)
    valid_pt = tf.concat([valid_coord, tf.gather_nd(rgb, idx)], axis=-1)

    return valid_pt
'''

def save_point_cloud(epoch):
    all_points = []
    batch_flat = []
    batch_t = []
    for i in range(images.shape[0]) :
        ray_oris, ray_dirs = get_rays(H, W, focal, tf.convert_to_tensor(poses[i], dtype=tf.float32))

        rays_flat, t_vals = render_flat_rays(
            ray_oris, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
        )

        if i % BATCH_SIZE == 0 and i > 0:
            batched_flat = tf.stack(batch_flat, axis=0)
            batch_flat = [rays_flat]

            batched_t = tf.stack(batch_t, axis=0)
            batch_t = [t_vals]

            rgb, depth = render_rgb_depth(
                nerf_model, batched_flat, batched_t, rand=False, train=False
            )
            print(i)
            valid_pt = depth_to_point(rgb, depth, ray_oris, ray_dirs)
            all_points.extend(valid_pt)

        else:
            batch_flat.append(rays_flat)
            batch_t.append(t_vals)

    # all_points를 numpy 배열로 변환
    all_points_array = np.array(all_points).reshape(-1, 6)

    # 배열을 좌표와 색상으로 분리
    coordinates = all_points_array[:, :3]
    colors = all_points_array[:, 3:]

    # DataFrame 생성
    df = pd.DataFrame(np.hstack((coordinates, colors)), columns=['x', 'y', 'z', 'r', 'g', 'b'])
    filename = str(epoch) + "_pointcloud"
    # CSV 파일로 저장
    df.to_csv(filename + '.csv', index=False)

class NeRF(keras.Model):
    def __init__(self, nerf_model):
        super().__init__()
        self.nerf_model = nerf_model
        

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="psnr")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    def train_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        with tf.GradientTape() as tape:
            # Get the predictions from the model.
            rgb, _ = render_rgb_depth(
                model=self.nerf_model, rays_flat=rays_flat, t_vals=t_vals, rand=True
            )
            loss = self.loss_fn(images, rgb) + sum(self.nerf_model.losses)

        # Get the trainable variables.
        trainable_variables = self.nerf_model.trainable_variables

        # Get the gradeints of the trainiable variables with respect to the loss.
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nerf_model.trainable_variables))

        # Apply the grads and optimize the model.
        # self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)
        

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.vq_loss_tracker.update_state(sum(self.nerf_model.losses))
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    def test_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        # Get the predictions from the model.
        rgb, _ = render_rgb_depth(
            model=self.nerf_model, rays_flat=rays_flat, t_vals=t_vals, rand=True
        )
        loss = self.loss_fn(images, rgb)

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_metric]


test_imgs, test_rays = next(iter(train_ds))
test_rays_flat, test_t_vals = test_rays

loss_list = []


class TrainMonitor(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        # Plot the rgb, depth and the loss plot.
        

        loss_list.append(loss)
        test_recons_images, depth_maps = render_rgb_depth(
            model=self.model.nerf_model,
            rays_flat=test_rays_flat,
            t_vals=test_t_vals,
            rand=True,
            train=False,
        )
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(keras.utils.array_to_img(test_recons_images[0]))
        ax[0].set_title(f"Predicted Image: {epoch:03d}")

        ax[1].imshow(keras.utils.array_to_img(depth_maps[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")

        
        plt.ylim(0,0.09)
        ax[2].plot(loss_list)
        #plt.xlim(0,EPOCHS+1)
        #ax[2].set_xticks(np.arange(0, EPOCHS + 1, 10.0))
        # ax[2].set_yticks(i for i in range(0,0.1,0.01))
        ax[2].set_xticks(np.arange(0, EPOCHS + 1, 10.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        fig.savefig(f"images/{epoch:03d}.png")

        # Save the model
        if epoch % save_interval == 0:  # 일정 주기로 모델 저장
            self.model.nerf_model.save('nerf_model.h5')

        if epoch % 50 == 0:
            save_point_cloud(epoch)
        
    def on_train_end(self, logs=None):
        self.end_time = time.time()
        print('Total training time is', self.end_time - self.start_time, 'seconds.')
        with open('loss_list.json', 'w') as f:
            json.dump(loss_list, f)


num_pos = H * W * NUM_SAMPLES
nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos)

model = NeRF(nerf_model)
model.compile(
    optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError()
)

# Create a directory to save the images during training.
if not os.path.exists("images"):
    os.makedirs("images")

model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[TrainMonitor()],
)

def create_gif(path_to_images, name_gif):
    filenames = glob.glob(path_to_images)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    kargs = {"duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)


create_gif("images/*.png", "training.gif")

# Get the trained NeRF model and infer.
nerf_model = model.nerf_model
test_recons_images, depth_maps = render_rgb_depth(
    model=nerf_model,
    rays_flat=test_rays_flat,
    t_vals=test_t_vals,
    rand=True,
    train=False,
)

def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, tf.cos(phi), -tf.sin(phi), 0],
        [0, tf.sin(phi), tf.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [tf.cos(theta), 0, -tf.sin(theta), 0],
        [0, 1, 0, 0],
        [tf.sin(theta), 0, tf.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w



'''
for i in range(images.shape[0]) :
    print(i)
    ray_oris, ray_dirs = get_rays(H, W, focal, tf.convert_to_tensor(poses[i], dtype=tf.float32))
    rays_flat, t_vals = render_flat_rays(
        ray_oris, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
    )

    if i % BATCH_SIZE == 0 and i > 0:
        batched_flat = tf.stack(batch_flat, axis=0)
        batch_flat = [rays_flat]

        batched_t = tf.stack(batch_t, axis=0)
        batch_t = [t_vals]

        rgb, depth = render_rgb_depth(
            nerf_model, batched_flat, batched_t, rand=False, train=False
        )

        valid_pt = depth_to_point(rgb, depth, ray_oris, ray_dirs)

        # 각 프레임의 포인트 클라우드 데이터를 all_points 리스트에 추가합니다.
        all_points.extend(valid_pt.numpy())  # valid_pt를 numpy 배열로 변환

    else:
        batch_flat.append(rays_flat)
        batch_t.append(t_vals)

'''

''' version 1
for i in range(images.shape[0]) :
   
    ray_oris, ray_dirs = get_rays(H, W, focal, tf.convert_to_tensor(poses[i], dtype=tf.float32))
    rays_flat, t_vals = render_flat_rays(
        ray_oris, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
    )

    batched_flat = tf.stack(batch_flat, axis=0)
    batch_flat = [rays_flat]

    batched_t = tf.stack(batch_t, axis=0)
    batch_t = [t_vals]

    rgb, depth = render_rgb_depth(
        nerf_model, batched_flat, batched_t, rand=False, train=False
    )

    valid_pt = depth_to_point(rgb, depth, ray_oris, ray_dirs)

    # 각 프레임의 포인트 클라우드 데이터를 all_points 리스트에 추가합니다.
    all_points.extend(valid_pt.numpy())  # valid_pt를 numpy 배열로 변환
'''

''' version 2
for i in range(images.shape[0]) :
    print("c2w : ", tf.convert_to_tensor(poses[i], dtype=tf.float32))
    ray_oris, ray_dirs = get_rays(H, W, focal, tf.convert_to_tensor(poses[i], dtype=tf.float32))
    print("ray_oris : ", ray_oris)
    print("ray_dirs : ", ray_dirs)
    rays_flat, t_vals = render_flat_rays(
        ray_oris, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
    )
    print("rays_flat : ", rays_flat)
    print("t_vals : ", t_vals)

    rgb, depth = render_rgb_depth(
            nerf_model, rays_flat, t_vals, rand=False, train=False
        )

    valid_pt = depth_to_point(rgb, depth, ray_oris, ray_dirs)

    # 각 프레임의 포인트 클라우드 데이터를 all_points 리스트에 추가합니다.
    all_points.extend(valid_pt.numpy())  # valid_pt를 numpy 배열로 변환
'''




'''
# all_points_array = np.array(all_points)
# all_points_reshaped = np.reshape(all_points_array, (-1, all_points_array.shape[-1]))
# np.savetxt('pointcloud.txt', all_points_reshaped)
# np.save('pointcloud', all_points)
# all_points_df = pd.DataFrame(all_points)
# all_points_df.to_csv("cloudcompare_input.csv", index=False, sep=',')
#all_points.to_csv("cloudcompare_input.csv", index_col = False, sep=',')
'''

rgb_frames = []
batch_flat = []
batch_t = []

# Iterate over different theta value and generate scenes.
for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False))):
    # Get the camera to world matrix.
    c2w = pose_spherical(theta, -30.0, 4.0)
    
    #
    ray_oris, ray_dirs = get_rays(H, W, focal, c2w)
    rays_flat, t_vals = render_flat_rays(
        ray_oris, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
    )

    if index % BATCH_SIZE == 0 and index > 0:
        batched_flat = tf.stack(batch_flat, axis=0)
        batch_flat = [rays_flat]

        batched_t = tf.stack(batch_t, axis=0)
        batch_t = [t_vals]

        rgb, depth = render_rgb_depth(
            nerf_model, batched_flat, batched_t, rand=False, train=False
        )

        temp_rgb = [np.clip(255 * img, 0.0, 255.0).astype(np.uint8) for img in rgb]

        rgb_frames = rgb_frames + temp_rgb

    else:
        batch_flat.append(rays_flat)
        batch_t.append(t_vals)


rgb_video = "rgb_video.mp4"
imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)