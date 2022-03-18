from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)

import numpy as np
import torch


# Basic data loading
def dataset_from_config(
    cfg,
    ):
    dataset = []

    for cam_idx, cam_key in enumerate(cfg.cameras.keys()):
        cam_cfg = cfg.cameras[cam_key]

        # Create camera parameters
        R, T = look_at_view_transform(
            eye=(cam_cfg.eye,),
            at=(cam_cfg.scene_center,),
            up=(cam_cfg.up,),
        )
        focal = torch.tensor([cam_cfg.focal])[None]
        principal_point = torch.tensor(cam_cfg.principal_point)[None]

        # Assemble the dataset
        image = None
        if 'image' in cam_cfg and cam_cfg.image is not None:
            image = torch.tensor(np.load(cam_cfg.image))[None]

        dataset.append(
            {
                "image": image,
                "camera": PerspectiveCameras(
                    focal_length=focal,
                    principal_point=principal_point,
                    R=R,
                    T=T,
                ),
                "camera_idx": cam_idx,
            }
        )

    return dataset


# Spiral cameras looking at the origin
def create_surround_cameras(radius, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=1.0):
    cameras = []

    for theta in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:

        if np.abs(up[1]) > 0:
            eye = [np.cos(theta + np.pi / 2) * radius, 1.0, -np.sin(theta + np.pi / 2) * radius]
        else:
            eye = [np.cos(theta + np.pi / 2) * radius, np.sin(theta + np.pi / 2) * radius, 2.0]

        R, T = look_at_view_transform(
            eye=(eye,),
            at=([0.0, 0.0, 0.0],),
            up=(up,),
        )

        cameras.append(
            PerspectiveCameras(
                focal_length=torch.tensor([focal_length])[None],
                principal_point=torch.tensor([0.0, 0.0])[None],
                R=R,
                T=T,
            )
        )
    
    return cameras


def vis_grid(xy_grid, image_size):
    xy_vis = (xy_grid + 1) / 2.001
    xy_vis = torch.cat([xy_vis, torch.zeros_like(xy_vis[..., :1])], -1)
    xy_vis = xy_vis.view(image_size[1], image_size[0], 3)
    xy_vis = np.array(xy_vis.detach().cpu())

    return xy_vis


def vis_rays(ray_bundle, image_size):
    rays = torch.abs(ray_bundle.directions)
    rays = rays.view(image_size[1], image_size[0], 3)
    rays = np.array(rays.detach().cpu())

    return rays