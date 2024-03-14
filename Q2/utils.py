import math
from typing import List

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    look_at_view_transform,
)
from skimage import img_as_ubyte
from torch.optim.lr_scheduler import LambdaLR


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, -1)


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
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def get_mesh_renderer_soft(image_size=512, lights=None, device=None, sigma=1e-4):
    """
    Create a soft renderer for differentaible texture rendering.
    Ref: https://pytorch3d.org/tutorials/fit_textured_mesh#3.-Mesh-and-texture-prediction-via-textured-rendering

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

    # Rasterization settings for differentiable rendering, where the blur_radius
    # initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable
    # Renderer for Image-based 3D Reasoning', ICCV 2019
    raster_settings_soft = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=50,
        perspective_correct=False,
    )

    # Differentiable soft renderer using per vertex RGB colors for texture
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings_soft),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def render_360_views(mesh, renderer, device, dist=3, elev=0, output_path=None):
    images = []
    for azim in range(0, 360, 10):
        R, T = look_at_view_transform(dist, elev, azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Place a point light in front of the cow.
        lights = PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        images.append(rend)

    # convert to uint8 to suppress "lossy conversion" warning
    images = [np.clip(img, -1, 1) for img in images]
    images = [img_as_ubyte(img) for img in images]

    # save a gif of the 360 rotation
    imageio.mimsave(output_path, images, fps=15)


from pytorch3d.io import load_obj, load_objs_as_meshes

def init_mesh(
    model_path,
    device="cpu",
):
    print("=> loading target mesh...")
    verts, faces, aux = load_obj(
        model_path, device=device, load_textures=True, create_texture_atlas=True
    )
    mesh = load_objs_as_meshes([model_path], device=device)
    faces = faces.verts_idx
    return mesh, verts, faces, aux


# calculate the text embs.
@torch.no_grad()
def prepare_embeddings(sds, prompt, neg_prompt="", view_dependent=False):
    # text embeddings (stable-diffusion)
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(neg_prompt, str):
        neg_prompt = [neg_prompt]
    embeddings = {}
    embeddings["default"] = sds.get_text_embeddings(prompt)  # shape [1, 77, 1024]
    embeddings["uncond"] = sds.get_text_embeddings(neg_prompt)  # shape [1, 77, 1024]
    if view_dependent:
        for d in ["front", "side", "back"]:
            embeddings[d] = sds.get_text_embeddings([f"{prompt}, {d} view"])
    return embeddings
