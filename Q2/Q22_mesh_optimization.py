import argparse
import os
import os.path as osp
import time

import numpy as np
import pytorch3d
import torch
from implicit import ColorField
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    TexturesVertex,
    look_at_view_transform,
)
from SDS import SDS
from tqdm import tqdm
from utils import (
    get_cosine_schedule_with_warmup,
    get_mesh_renderer_soft,
    init_mesh,
    prepare_embeddings,
    render_360_views,
    seed_everything,
)


def optimize_mesh_texture(
    sds,
    mesh_path,
    prompt,
    neg_prompt="",
    device="cpu",
    log_interval=100,
    save_mesh=True,
    args=None,
):
    """
    Optimize the texture map of a mesh to match the prompt.
    """
    # Step 1. Create text embeddings from prompt
    embeddings = prepare_embeddings(sds, prompt, neg_prompt, view_dependent=False)
    sds.text_encoder.to("cpu")  # free up GPU memory
    torch.cuda.empty_cache()

    # Step 2. Load the mesh
    mesh, vertices, faces, aux = init_mesh(mesh_path, device=device)
    vertices = vertices.unsqueeze(0).to(device)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0).to(device)  # (N_f, 3) -> (1, N_f, 3)

    # Step 2.1 Initialize a randome texture map (optimizable parameter)
    # create a texture field with implicit function
    color_field = ColorField().to(device)  # input (1, N_v, xyz) -> output (1, N_v, rgb)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=TexturesVertex(verts_features=color_field(vertices)),
    )
    mesh = mesh.to(device)

    # Step 3.1 Initialize the renderer
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    renderer = get_mesh_renderer_soft(image_size=512, device=device, lights=lights)

    # For logging purpose, render 360 views of the initial mesh
    if save_mesh:
        render_360_views(
            mesh.detach(),
            renderer,
            device=device,
            output_path=osp.join(sds.output_dir, "initial_mesh.gif"),
        )

    # Step 3.2. Initialize the cameras
    # check the size of the mesh so that it is in the field of view
    print(
        f"check mesh range: {vertices.min()}, {vertices.max()}, center {vertices.mean(1)}"
    )

    ### YOUR CODE HERE ###
    # create a list of query cameras as the training set
    # Note: to create the dataset, you can either pre-define a list of query cameras as below or randomly sample a camera pose on the fly in the training loop.
    query_cameras = [] # optional

    # Step 4. Create optimizer training parameters
    optimizer = torch.optim.AdamW(color_field.parameters(), lr=5e-4, weight_decay=0)
    total_iter = 2000
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(total_iter * 1.5))

    # Step 5. Training loop to optimize the texture map
    loss_dict = {}
    for i in tqdm(range(total_iter)):
        # Initialize optimizer
        optimizer.zero_grad()

        # Update the textures
        mesh.textures = TexturesVertex(verts_features=color_field(vertices))

        ### YOUR CODE HERE ###

        # Forward pass
        # Render a randomly sampled camera view to optimize in this iteration
        rend = 
        # Encode the rendered image to latents
        latents = 
        # Compute the loss
        loss =



        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # clamping the latents to avoid over saturation
        latents.data = latents.data.clip(-1, 1)
        if i % log_interval == 0 or i == total_iter - 1:
            # save the loss
            loss_dict[i] = loss.item()

            # save the image
            img = sds.decode_latents(latents.detach())
            output_im = Image.fromarray(img.astype("uint8"))
            output_path = os.path.join(
                sds.output_dir,
                f"output_{prompt[0].replace(' ', '_')}_iter_{i}.png",
            )
            output_im.save(output_path)

    if save_mesh:
        render_360_views(
            mesh.detach(),
            renderer,
            device=device,
            output_path=osp.join(sds.output_dir, f"final_mesh.gif"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a hamburger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="postfix for the output directory to differentiate multiple runs",
    )

    parser.add_argument(
        "-m",
        "--mesh_path",
        type=str,
        default="data/cow.obj",
        help="Path to the input image",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    # create output directory
    args.output_dir = osp.join(args.output_dir, "mesh")
    output_dir = os.path.join(
        args.output_dir, args.prompt.replace(" ", "_") + args.postfix
    )
    os.makedirs(output_dir, exist_ok=True)

    # initialize SDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sds = SDS(sd_version="2.1", device=device, output_dir=output_dir)

    # optimize the texture map of a mesh
    start_time = time.time()
    assert (
        args.mesh_path is not None
    ), "mesh_path should be provided for optimizing the texture map for a mesh"
    optimize_mesh_texture(
        sds, mesh_path=args.mesh_path, prompt=args.prompt, device=device, args=args
    )
    print(f"Optimization took {time.time() - start_time:.2f} seconds")
