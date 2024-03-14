import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians

from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform

def run_test_1():
    """
    Tests cov_3D computation for isotropic Gaussians
    """
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device="cpu", isotropic=True
    )

    scales = torch.tensor([
        [ 1.0057],
        [ 0.5317],
        [ 1.2069],
        [ 2.0091],
        [-1.0004]
    ]).to(torch.float32)

    quats = torch.tensor([
        [ 0.3243, -0.3977,  0.7983, -0.3153],
        [ 0.5657,  0.2395, -0.5910,  0.5228],
        [ 0.2598, -0.7521, -0.3487,  0.4951],
        [ 0.3730,  0.7351, -0.5261,  0.2089],
        [ 0.4079,  0.4792,  0.0567,  0.7751]
    ]).to(torch.float32)

    gt_answer = torch.tensor([
        [[1.0114, 0.0000, 0.0000],
         [0.0000, 1.0114, 0.0000],
         [0.0000, 0.0000, 1.0114]],

        [[0.2827, 0.0000, 0.0000],
         [0.0000, 0.2827, 0.0000],
         [0.0000, 0.0000, 0.2827]],

        [[1.4566, 0.0000, 0.0000],
         [0.0000, 1.4566, 0.0000],
         [0.0000, 0.0000, 1.4566]],

        [[4.0365, 0.0000, 0.0000],
         [0.0000, 4.0365, 0.0000],
         [0.0000, 0.0000, 4.0365]],

        [[1.0008, 0.0000, 0.0000],
         [0.0000, 1.0008, 0.0000],
         [0.0000, 0.0000, 1.0008]]
    ]).to(torch.float32)

    your_answer = gaussians.compute_cov_3D(quats, scales)

    return torch.all(torch.isclose(your_answer, gt_answer, rtol=0.0, atol=1e-3))

def run_test_2():
    """
    Tests cov_2D computation for anisotropic Gaussians
    """
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device="cpu", isotropic=False
    )

    means_3D = torch.tensor([
        [ 0.4123, -0.0350,  1.0627],
        [ 0.1975, -0.5187, -0.0357],
        [ 0.2168, -1.2460,  0.7319],
        [ 0.4533,  0.5670, -0.3811],
        [-1.1483, -2.0718, -0.7977]
    ]).to(torch.float32)

    scales = torch.tensor([
        [ 0.1642,  2.4228, -0.0908],
        [ 0.9635,  1.1696,  0.3278],
        [-0.6131,  1.0911, -0.9228],
        [-0.1720, -0.5747, -0.6813],
        [ 0.1222,  1.7187, -1.5246]
    ]).to(torch.float32)

    quats = torch.tensor([
        [ 0.3243, -0.3977,  0.7983, -0.3153],
        [ 0.5657,  0.2395, -0.5910,  0.5228],
        [ 0.2598, -0.7521, -0.3487,  0.4951],
        [ 0.3730,  0.7351, -0.5261,  0.2089],
        [ 0.4079,  0.4792,  0.0567,  0.7751]
    ]).to(torch.float32)

    dist = 6.0
    dim = 128
    img_size = (dim, dim)
    R, T = look_at_view_transform(dist = dist, azim=75.0, elev=45.0)
    camera = PerspectiveCameras(
        focal_length=5.0 * dim/2.0, in_ndc=False,
        principal_point=((dim/2, dim/2),),
        R=R, T=T, image_size=(img_size,),
    ).to("cpu")

    gt_answer = torch.tensor([
        [[ 2618.8611,  5941.3306],
         [ 5941.3301, 14066.5830]],

        [[ 1762.7010,  1096.4006],
         [ 1096.4005,  1337.9269]],

        [[ 2478.8181,   339.9595],
         [  339.9595,   992.3395]],

        [[ 1645.8475,  -170.6580],
         [ -170.6579,   233.4286]],

        [[  623.9609,  -285.9298],
         [ -285.9297,  4030.9944]]
    ]).to(torch.float32)

    your_answer = gaussians.compute_cov_2D(means_3D, quats, scales, camera, img_size)

    return torch.all(torch.isclose(your_answer, gt_answer, rtol=1e-4, atol=1e-8))

def run_test_3():
    """
    Tests means_2D computation for isotropic Gaussians
    """
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device="cpu", isotropic=True
    )

    means_3D = torch.tensor([
        [ 1.0226,  1.3136, -1.1826],
        [-0.7536, -0.5747, -0.5632],
        [-0.4628, -0.9475,  0.4862],
        [-0.2915,  3.0803,  0.1183],
        [-0.9359,  1.7944, -0.5683]
    ]).to(torch.float32)

    dist = 6.0
    dim = 128
    img_size = (dim, dim)
    R, T = look_at_view_transform(dist = dist, azim=75.0, elev=45.0)
    camera = PerspectiveCameras(
        focal_length=5.0 * dim/2.0, in_ndc=False,
        principal_point=((dim/2, dim/2),),
        R=R, T=T, image_size=(img_size,),
    ).to("cpu")

    gt_answer = torch.tensor([
        [ 162.1082,   32.8420],
        [  79.8977,   54.3685],
        [  36.6533,   84.5473],
        [  48.8203, -124.4747],
        [  81.9282,  -53.6132]
    ]).to(torch.float32)

    your_answer = gaussians.compute_means_2D(means_3D, camera)

    return torch.all(torch.isclose(your_answer, gt_answer, rtol=1e-4, atol=1e-8))

def run_test_4():
    """
    Tests evaluate_gaussian_2D computation for anisotropic Gaussians
    """
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device="cpu", isotropic=False
    )

    means_3D = torch.tensor([
        [ 0.4123, -0.0350,  1.0627],
        [ 0.1975, -0.5187, -0.0357],
        [ 0.2168, -1.2460,  0.7319],
        [ 0.4533,  0.5670, -0.3811],
        [-1.1483, -2.0718, -0.7977]
    ]).to(torch.float32)

    scales = torch.tensor([
        [ 0.1642,  2.4228, -0.0908],
        [ 0.9635,  1.1696,  0.3278],
        [-0.6131,  1.0911, -0.9228],
        [-0.1720, -0.5747, -0.6813],
        [ 0.1222,  1.7187, -1.5246]
    ]).to(torch.float32)

    quats = torch.tensor([
        [ 0.3243, -0.3977,  0.7983, -0.3153],
        [ 0.5657,  0.2395, -0.5910,  0.5228],
        [ 0.2598, -0.7521, -0.3487,  0.4951],
        [ 0.3730,  0.7351, -0.5261,  0.2089],
        [ 0.4079,  0.4792,  0.0567,  0.7751]
    ]).to(torch.float32)

    dist = 6.0
    dim = 128
    img_size = (dim, dim)
    R, T = look_at_view_transform(dist = dist, azim=75.0, elev=45.0)
    camera = PerspectiveCameras(
        focal_length=5.0 * dim/2.0, in_ndc=False,
        principal_point=((dim/2, dim/2),),
        R=R, T=T, image_size=(img_size,),
    ).to("cpu")

    gt_answer = torch.tensor([
        [ -3.9607,  -4.2238,  -4.4959],
        [ -3.0004,  -3.0066,  -3.0140],
        [ -7.3205,  -7.3244,  -7.3288],
        [-12.1272, -12.0402, -11.9539],
        [ -7.3011,  -7.1561,  -7.0127]
    ]).to(torch.float32)

    cov_2D = gaussians.compute_cov_2D(means_3D, quats, scales, camera, img_size)
    means_2D = gaussians.compute_means_2D(means_3D, camera)

    W, H = img_size
    xs, ys = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    points_2D = torch.stack((xs.flatten(), ys.flatten()), dim = 1)  # (H*W, 2)
    points_2D = points_2D[:3].to("cpu")

    points_2D = points_2D.unsqueeze(0)
    means_2D = means_2D.unsqueeze(1)

    cov_2D_inverse = Gaussians.invert_cov_2D(cov_2D)
    your_answer = Gaussians.evaluate_gaussian_2D(points_2D, means_2D, cov_2D_inverse)

    return torch.all(torch.isclose(your_answer, gt_answer, rtol=1e-4, atol=1e-8))

def run_tests():

    counter = 0
    total = 4

    if run_test_1():
        counter += 1
    else:
        print("Test 1 Fail!")

    if run_test_2():
        counter += 1
    else:
        print("Test 2 Fail!")

    if run_test_3():
        counter += 1
    else:
        print("Test 3 Fail!")

    if run_test_4():
        counter += 1
    else:
        print("Test 4 Fail!")

    print(f"[{counter}/{total}] Tests Passed")

if __name__ == "__main__":

    run_tests()
