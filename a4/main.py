import os
import warnings

import hydra
import numpy as np
import torch
import tqdm
import imageio

from omegaconf import DictConfig
from PIL import Image
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
import matplotlib.pyplot as plt

from sampler import sampler_dict
from a4.implicit import implicit_dict
from a4.renderer import renderer_dict
from a4.losses import eikonal_loss, sphere_loss, get_random_points, select_random_points

from ray_utils import (
    sample_images_at_xy,
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_rays_from_pixels
)
from data_utils import (
    dataset_from_config,
    create_surround_cameras,
    vis_grid,
    vis_rays,
)
from dataset import (
    get_nerf_datasets,
    trivial_collate,
)
from render_functions import render_geometry
from render_functions import render_points


# Model class containing:
#   1) Implicit function defining the scene
#   2) Sampling scheme which generates sample points along rays
#   3) Renderer which can render an implicit function given a sampling scheme

class Model(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        # Get implicit function from config
        self.implicit_fn = implicit_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )

        # Point sampling (raymarching) scheme
        self.sampler = sampler_dict[cfg.sampler.type](
            cfg.sampler
        )

        # Initialize implicit renderer
        self.renderer = renderer_dict[cfg.renderer.type](
            cfg.renderer
        )
    
    def forward(
        self,
        ray_bundle
    ):
        # Call renderer with
        #  a) Implicit function
        #  b) Sampling routine

        return self.renderer(
            self.sampler,
            self.implicit_fn,
            ray_bundle
        )


def render_images(
    model,
    cameras,
    image_size,
    save=False,
    file_prefix=''
):
    all_images = []
    device = list(model.parameters())[0].device

    for cam_idx, camera in enumerate(cameras):
        print(f'Rendering image {cam_idx}')

        with torch.no_grad():
            torch.cuda.empty_cache()

            # Get rays
            camera = camera.to(device)
            xy_grid = get_pixels_from_image(image_size, camera)
            ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)

            # Run model forward
            out = model(ray_bundle)

        # Return rendered features (colors)
        image = np.array(
            out['color'].view(
                image_size[1], image_size[0], 3
            ).detach().cpu()
        )
        all_images.append(image)

        # Save
        if save:
            plt.imsave(
                f'{file_prefix}_{cam_idx}.png',
                image
            )
    
    return all_images


def render(
    cfg,
):
    # Create model
    model = Model(cfg)
    model = model.cuda(); model.eval()

    # Render spiral
    cameras = create_surround_cameras(3.0, n_poses=20, up=(0.0, 0.0, 1.0))
    all_images = render_images(
        model, cameras, cfg.data.image_size
    )
    imageio.mimsave('images/part_1.gif', [np.uint8(im * 255) for im in all_images])


def create_model(cfg):
    # Create model
    model = Model(cfg)
    model.cuda(); model.train()

    # Load checkpoints
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.training.checkpoint_path
    )

    if len(cfg.training.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.training.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return cfg.training.lr_scheduler_gamma ** (
            epoch / cfg.training.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    return model, optimizer, lr_scheduler, start_epoch, checkpoint_path


def train_points(
    cfg
):
    # Create model
    model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg)

    # Pretrain SDF
    pretrain_sdf(cfg, model)

    # Load pointcloud
    point_cloud = np.load(cfg.data.point_cloud_path)
    all_points = torch.Tensor(point_cloud["verts"][::2]).cuda().view(-1, 3)
    all_points = all_points - torch.mean(all_points, dim=0).unsqueeze(0)
    
    point_images = render_points(
        all_points.unsqueeze(0), create_surround_cameras(3.0, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=2.0),
        cfg.data.image_size, file_prefix='points'
    )
    imageio.mimsave('images/part_2_input.gif', [np.uint8(im * 255) for im in point_images])

    # Run the main training loop.
    for epoch in range(0, cfg.training.num_epochs):
        t_range = tqdm.tqdm(range(0, all_points.shape[0], cfg.training.batch_size))

        for idx in t_range:
            # Select random points from pointcloud
            points = select_random_points(all_points, cfg.training.batch_size)

            # Get distances and enforce point cloud loss
            distances, gradients = model.implicit_fn.get_distance_and_gradient(points)
            loss = None # TODO (Q2): Point cloud SDF loss on distances
            point_loss = loss

            # Sample random points in bounding box
            eikonal_points = get_random_points(
                cfg.training.batch_size, cfg.training.bounds, 'cuda'
            )

            # Get sdf gradients and enforce eikonal loss
            eikonal_distances, eikonal_gradients = model.implicit_fn.get_distance_and_gradient(eikonal_points)
            loss += torch.exp(-1e2 * torch.abs(eikonal_distances)).mean() * cfg.training.inter_weight
            loss += eikonal_loss(eikonal_gradients) * cfg.training.eikonal_weight # TODO (Q2): Implement eikonal loss

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {point_loss:.06f}')
            t_range.refresh()

        # Checkpoint.
        if (
            epoch % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        # Render
        if (
            epoch % cfg.training.render_interval == 0
            and epoch > 0
        ):
            try:
                test_images = render_geometry(
                    model, create_surround_cameras(3.0, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=2.0),
                    cfg.data.image_size, file_prefix='eikonal', thresh=0.002,
                )
                imageio.mimsave('images/part_2.gif', [np.uint8(im * 255) for im in test_images])
            except Exception as e:
                print("Empty mesh")
                pass


def pretrain_sdf(
    cfg,
    model
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Run the main training loop.
    for iter in range(0, cfg.training.pretrain_iters):
        points = get_random_points(
            cfg.training.batch_size, cfg.training.bounds, 'cuda'
        )

        # Run model forward
        distances = model.implicit_fn.get_distance(points)
        loss = sphere_loss(distances, points, 1.0)

        # Take the training step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_images(
    cfg
):
    # Create model
    model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg)

    # Load the training/validation data.
    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    # Pretrain SDF
    pretrain_sdf(cfg, model)

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            image, camera, camera_idx = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, cfg.data.image_size, camera
            )
            ray_bundle = get_rays_from_pixels(
                xy_grid, cfg.data.image_size, camera
            )
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Run model forward
            out = model(ray_bundle)

            # Color loss
            loss = torch.mean(torch.square(rgb_gt - out['color']))
            image_loss = loss

            # Sample random points in bounding box
            eikonal_points = get_random_points(
                cfg.training.batch_size, cfg.training.bounds, 'cuda'
            )

            # Get sdf gradients and enforce eikonal loss
            eikonal_distances, eikonal_gradients = model.implicit_fn.get_distance_and_gradient(eikonal_points)
            loss += torch.exp(-1e2 * torch.abs(eikonal_distances)).mean() * cfg.training.inter_weight
            loss += eikonal_loss(eikonal_gradients) * cfg.training.eikonal_weight # TODO (2): Implement eikonal loss

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {image_loss:.06f}')
            t_range.refresh()

        # Adjust the learning rate.
        lr_scheduler.step()

        # Checkpoint.
        if (
            epoch % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        # Render
        if (
            epoch % cfg.training.render_interval == 0
            and epoch > 0
        ):
            test_images = render_images(
                model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
                cfg.data.image_size, file_prefix='volsdf'
            )
            imageio.mimsave('images/part_3.gif', [np.uint8(im * 255) for im in test_images])

            try:
                test_images = render_geometry(
                    model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
                    cfg.data.image_size, file_prefix='volsdf_geometry'
                )
                imageio.mimsave('images/part_3_geometry.gif', [np.uint8(im * 255) for im in test_images])
            except Exception as e:
                print("Empty mesh")
                pass
                

@hydra.main(config_path='configs', config_name='torus')
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.type == 'render':
        render(cfg)
    elif cfg.type == 'train_points':
        train_points(cfg)
    elif cfg.type == 'train_images':
        train_images(cfg)


if __name__ == "__main__":
    main()


