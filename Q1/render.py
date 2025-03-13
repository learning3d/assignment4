import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Gaussians, Scene
from data_utils import colour_depth_q1_render
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform

def create_renders(args):

    dim = args.img_dim
    img_size = (dim, dim)

    num_views = 32
    azims = np.linspace(-180, 180, num_views)
    elevs = np.linspace(-180, 180, num_views)

    debug_root = os.path.join(args.out_path, "q1_render")
    if not os.path.exists(debug_root):
        os.makedirs(debug_root, exist_ok=True)

    # Loading pre-trained gaussians
    gaussians = Gaussians(
        load_path=args.data_path, init_type="gaussians",
        device=args.device
    )

    # Preprocessing for ease of rendering
    new_points = gaussians.means - gaussians.means.mean(dim=0, keepdims=True)
    gaussians.means = new_points

    # Creating the scene with the loaded gaussians
    scene = Scene(gaussians)

    imgs = []
    for i in tqdm(range(num_views), desc="Rendering"):

        dist = 6.0
        R, T = look_at_view_transform(dist = dist, azim=azims[i], elev=elevs[i], up=((0, -1, 0),))
        camera = PerspectiveCameras(
            focal_length=5.0 * dim/2.0, in_ndc=False,
            principal_point=((dim/2, dim/2),),
            R=R, T=T, image_size=(img_size,),
        ).to(args.device)

        with torch.no_grad():
            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (1.0, 1.0, 1.0)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: img_size and camera are available above
            img, depth, mask = None

        debug_path = os.path.join(debug_root, f"{i:03d}.png")
        img = img.detach().cpu().numpy()
        mask = mask.repeat(1, 1, 3).detach().cpu().numpy()
        depth = depth.detach().cpu().numpy()

        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        mask = np.where(mask > 0.5, 255.0, 0.0).astype(np.uint8)  # (H, W, 3)

        # Colouring the depth map
        depth = depth[:, :, 0].astype(np.float32)  # (H, W)
        coloured_depth = colour_depth_q1_render(depth)  # (H, W, 3)

        concat = np.concatenate([img, coloured_depth, mask], axis = 1)
        resized = Image.fromarray(concat).resize((256*3, 256))
        resized.save(debug_path)

        imgs.append(np.array(resized))

    gif_path = os.path.join(args.out_path, "q1_render.gif")
    imageio.mimwrite(gif_path, imgs, duration=1000.0*(1/10.0), loop=0)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/chair.ply", type=str,
        help="Path to the pre-trained gaussian data to be rendered."
    )
    parser.add_argument(
        "--img_dim", default=256, type=int,
        help=(
            "Spatial dimension of the rendered image. "
            "The rendered image will have img_dim as its height and width."
        )
    )
    parser.add_argument(
        "--gaussians_per_splat", default=2048, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    create_renders(args)
