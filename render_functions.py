import argparse
import os
import sys
import datetime
import time
import math
import json
import torch

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import pytorch3d
import torch

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import OpenGLPerspectiveCameras
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)

import mcubes


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def render_points(
    points,
    cameras,
    image_size,
    save=False,
    file_prefix='',
    color=[0.7, 0.7, 1]
):
    device = points.device
    if device is None:
        device = get_device()

    # Get the renderer.
    points_renderer = get_points_renderer(image_size=image_size[0], radius=0.01)

    textures = torch.ones(points.size()).to(device)   # (1, N_v, 3)
    rgb = textures * torch.tensor(color).to(device)  # (1, N_v, 3)

    point_cloud = pytorch3d.structures.pointclouds.Pointclouds(
        points=points, features=rgb
    )
    
    all_images = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for cam_idx in range(len(cameras)):
            image = points_renderer(point_cloud, cameras=cameras[cam_idx].to(device))
            image = image[0,:,:,:3].detach().cpu().numpy()
            all_images.append(image)

            # Save
            if save:
                plt.imsave(
                    f'{file_prefix}_{cam_idx}.png',
                    image
                )
    
    return all_images



def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.
    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def implicit_to_mesh(implicit_fn, scale=0.5, grid_size=128, device='cpu', color=[0.7, 0.7, 1], chunk_size=262144, thresh=0):
    Xs = torch.linspace(-1*scale, scale, grid_size+1).to(device)
    Ys = torch.linspace(-1*scale, scale, grid_size+1).to(device)
    Zs = torch.linspace(-1*scale, scale, grid_size+1).to(device)
    grid = torch.stack(torch.meshgrid(Xs, Ys, Zs), dim=-1)

    grid = grid.view(-1, 3)
    num_points = grid.shape[0]
    sdfs = torch.zeros(num_points)
    
    with torch.no_grad():
        for chunk_start in range(0, num_points, chunk_size):
            torch.cuda.empty_cache()
            chunk_end = min(num_points, chunk_start+chunk_size)
            sdfs[chunk_start:chunk_end] = implicit_fn.get_distance(grid[chunk_start:chunk_end,:]).view(-1)

        sdfs = sdfs.view(grid_size+1, grid_size+1, grid_size+1)

    vertices, triangles = mcubes.marching_cubes(sdfs.cpu().numpy(), thresh)
    # normalize to [-scale, scale]
    vertices = (vertices/grid_size - 0.5)*2*scale

    vertices = torch.from_numpy(vertices).unsqueeze(0).float()
    faces = torch.from_numpy(triangles.astype(np.int64)).unsqueeze(0)
    
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    return mesh
    

def render_geometry(
    model,
    cameras,
    image_size,
    save=False,
    thresh=0.,
    file_prefix=''    
):
    device = list(model.parameters())[0].device
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    mesh_renderer = get_mesh_renderer(image_size=image_size[0], lights=lights, device=device)

    mesh = implicit_to_mesh(model.implicit_fn, scale=3, device=device, thresh=thresh)
    all_images = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for cam_idx in range(len(cameras)):
            image = mesh_renderer(mesh, cameras=cameras[cam_idx].to(device))
            image = image[0,:,:,:3].detach().cpu().numpy()
            all_images.append(image)

            # Save
            if save:
                plt.imsave(
                    f'{file_prefix}_{cam_idx}.png',
                    image
                )
    
    return all_images