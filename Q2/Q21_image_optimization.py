import argparse
import os
import os.path as osp
import time

import torch
import torch.nn as nn
from PIL import Image
from SDS import SDS
from tqdm import tqdm
from utils import get_cosine_schedule_with_warmup, prepare_embeddings, seed_everything


def optimize_an_image(
    sds,
    prompt,
    neg_prompt="",
    img=None,
    log_interval=100,
    args=None
):
    """
    Optimize an image to match the prompt.
    """
    # Step 1. Create text embeddings from prompt
    embeddings = prepare_embeddings(sds, prompt, neg_prompt, view_dependent=False)
    sds.text_encoder.to("cpu")  # free up GPU memory
    torch.cuda.empty_cache()

    # Step 2. Initialize latents to optimize
    latents = nn.Parameter(torch.randn(1, 4, 64, 64, device=sds.device))

    # Step 3. Create optimizer and loss function
    optimizer = torch.optim.AdamW([latents], lr=1e-1, weight_decay=0)
    total_iter = 2000
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(total_iter * 1.5))

    # Step 4. Training loop to optimize the latents
    for i in tqdm(range(total_iter)):
        optimizer.zero_grad()
        # Forward pass to compute the loss
        
        ### YOUR CODE HERE ###
        if args.sds_guidance:
            loss = 
        else:
            loss = 

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # clamping the latents to avoid over saturation
        latents.data = latents.data.clip(-1, 1)

        if i % log_interval == 0 or i == total_iter - 1:
            # Decode the image to visualize the progress
            img = sds.decode_latents(latents.detach())
            # Save the image
            output_im = Image.fromarray(img.astype("uint8"))
            output_path = os.path.join(
                sds.output_dir,
                f"output_{prompt[0].replace(' ', '_')}_iter_{i}.png",
            )
            output_im.save(output_path)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a hamburger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--sds_guidance", type=int, default=0, choices=[0, 1], help="boolen option to add guidance to the SDS loss")
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="postfix for the output directory to differentiate multiple runs",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    # create output directory
    args.output_dir = osp.join(args.output_dir, "image")
    output_dir = os.path.join(
        args.output_dir, args.prompt.replace(" ", "_") + args.postfix
    )
    os.makedirs(output_dir, exist_ok=True)

    # initialize SDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sds = SDS(sd_version="2.1", device=device, output_dir=output_dir)

    # optimize an image
    prompt = args.prompt
    start_time = time.time()
    img = optimize_an_image(sds, prompt=prompt, args=args)
    print(f"Optimization took {time.time() - start_time:.2f} seconds")

    # save the output image
    img = Image.fromarray(img.astype("uint8"))
    output_path = os.path.join(output_dir, f"output.png")
    print(f"Saving image to {output_path}")
    img.save(output_path)
