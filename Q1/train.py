import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import TruckDataset, visualize_renders
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    pass

def setup_optimizer(gaussians):

    gaussians.check_if_trainable()

    ### YOUR CODE HERE ###
    # HINT: Modify the learning rates to reasonable values. We have intentionally
    # set very high learning rates for all parameters.
    # HINT: Consider reducing the learning rates for parameters that seem to vary too
    # fast with the default settings.
    # HINT: Consider setting different learning rates for different sets of parameters.
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 0.05, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 0.05, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 0.05, "name": "colours"},
        {'params': [gaussians.means], 'lr': 0.05, "name": "means"},
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)
    optimizer = None

    return optimizer

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    # Setting up dataset
    train_dataset = TruckDataset(root=args.data_path, split="train")
    test_dataset = TruckDataset(root=args.data_path, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=TruckDataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=TruckDataset.collate_fn
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "q1_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "q1_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i][0]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [train_dataset[i][1].cuda() for i in viz_idxs]

    # Init gaussians and scene. Do note that we are setting isotropic to True
    gaussians = Gaussians(
        load_path=train_dataset.points_path, init_type="points",
        device=args.device, isotropic=True
    )
    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer = setup_optimizer(gaussians)

    # Training loop
    viz_frames = []
    for itr in range(args.num_itrs):

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img, camera, gt_mask = data

        gt_img = gt_img[0].cuda()
        camera = camera[0].cuda()
        if gt_mask is not None:
            gt_mask = gt_mask[0].cuda()

        # Rendering scene using gaussian splatting
        ### YOUR CODE HERE ###
        # HINT: Can any function from the Scene class help?
        # HINT: Set bg_colour to (0.0, 0.0, 0.0)
        # HINT: Get img_size from train_dataset
        # HINT: Get per_splat from args.gaussians_per_splat
        # HINT: camera is available above
        pred_img = None

        # Compute loss
        ### YOUR CODE HERE ###
        # HINT: A simple standard loss function should work.
        loss = None

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders(
                scene, gt_viz_img,
                viz_cameras, train_dataset.img_size
            )
            viz_frames.append(viz_frame)

    print("[*] Training Completed.")

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=TruckDataset.collate_fn
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img, camera, gt_mask = viz_data
        gt_img = gt_img[0].cuda()
        camera = camera[0].cuda()
        if gt_mask is not None:
            gt_mask = gt_mask[0].cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Get img_size from train_dataset
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img = None

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for test_data in tqdm(test_loader, desc="Running Evaluation"):

        gt_img, camera, gt_mask = test_data
        gt_img = gt_img[0].cuda()
        camera = camera[0].cuda()
        if gt_mask is not None:
            gt_mask = gt_mask[0].cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Get img_size from test_dataset
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img = None

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/truck", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
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
    parser.add_argument(
        "--num_itrs", default=1000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)
