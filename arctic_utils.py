import torch
import numpy as np
import warnings
import json
import os
import cv2
from PIL import Image
import glob
import imageio
import pathlib

SCENE_SCALE = 1.0

def quaternion_to_rotation_matrix(quaternions):
    """
    Converts a batch of quaternions to corresponding rotation matrices.

    Args:
        quaternions (np.ndarray): Array of shape (..., 4) representing quaternions.

    Returns:
        np.ndarray: Array of shape (..., 3, 3) representing rotation matrices.
    """
    quaternions = np.array(quaternions)
    assert quaternions.shape[-1] == 4, "Input quaternions must have shape (..., 4)"
    
    q = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute rotation matrix elements
    R = np.zeros((*q.shape[:-1], 3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    R[..., 0, 1] = 2 * (qx * qy - qz * qw)
    R[..., 0, 2] = 2 * (qx * qz + qy * qw)
    R[..., 1, 0] = 2 * (qx * qy + qz * qw)
    R[..., 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    R[..., 1, 2] = 2 * (qy * qz - qx * qw)
    R[..., 2, 0] = 2 * (qx * qz - qy * qw)
    R[..., 2, 1] = 2 * (qy * qz + qx * qw)
    R[..., 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)

    return R


def read_images_from_directory(dir :str, prefix = None, num_frames =10, start_frame = 12, sample_rate = 6):
    """
    Reads all images from a directory, optionally filtering by prefix.
    Supports PNG, JPG, JPEG, and TIFF formats.
    Returns a numpy array of stacked images.
    """
    print(f"Reading images from directory: {dir} with prefix: {prefix}")
    images = []
    if prefix is None:
        filenames = [os.path.join(dir, f"{i:05d}.jpg") for i in range(start_frame, start_frame+ num_frames*sample_rate, sample_rate)]
        for filepath in filenames:
            if os.path.isfile(filepath) and filepath.endswith(('.png', '.jpg', '.jpeg')):
                img = np.array(Image.open(filepath))
                images.append(img)

    else:
        for filename in sorted(os.listdir(dir)):
            if filename.startswith(prefix):
                filepath = os.path.join(dir, filename)
                if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')) and int(filepath.split('/')[-1][:-4]) <= start_frame + num_frames*sample_rate and int(filepath.split('/')[-1][:-4]) >= start_frame:
                    img = np.array(Image.open(filepath))
                    images.append(img)
                elif os.path.isfile(filepath) and filename.lower().endswith(('.tiff')):
                    img = np.asarray(imageio.v2.imread(pathlib.Path(filepath).read_bytes(), format="tiff"), dtype=np.float64)[:,:, 0]
                    images.append(img)

    return np.stack(images, axis=0)

def read_cam_arctic(sequence :str, idx :int, query_points = None, depth: str = 'dust3r', num_cams: int = 5):
    """
    Reads camera and depth data for a given sequence and view index.
    Selects the appropriate depth type: 'gt', 'vggt', or 'dust3r'.
    """
    if depth == 'dust3r':
        return read_cam_arctic_dust3r(sequence, idx, query_points, num_cams=num_cams)
    else:
        raise ValueError(f"Unknown depth type: {depth}. Use 'dust3r'.")
    
def get_cam_info(calib_file, subject, idx):
    """
    Load camera information from the calibration file.

    Args:
        calib_file (str): Path to the calibration file.

    Returns:
        dict: Dictionary containing camera parameters.
    """
    cam_info = {}

    with open(calib_file, 'r') as f:
        meta = json.load(f)

    scene_info = meta[subject]
    intr_matrix = np.array(scene_info['intris_mat'][idx])
    extr_matrix = np.array(scene_info['world2cam'][idx])
    extr_matrix[:3, 3] = extr_matrix[:3, 3] / SCENE_SCALE  # Scale translation vector
    cam_info['intrinsics'] = intr_matrix
    cam_info['extrinsics'] = extr_matrix



    return cam_info

def read_cam_arctic_dust3r(sequence :str, idx :str, depth_euclid = False, query_points = None, num_cams = 8):
    """
    Reads DUST3R-predicted depth, camera intrinsics, extrinsics, and images for a given view.
    Optionally interpolates depth at query points.
    """


    cam_ID = idx
    meta = get_cam_info(os.path.join(sequence, '../../..',  'meta', 'misc.json'), idx=int(idx)-1, subject='s01')
    intrinsics = meta['intrinsics'][:3, :3]
    extrinsics = meta['extrinsics']

    # Select label based on number of cameras
    duster_label = f'duster-views-{"".join(str(i) for i in range(num_cams))}'

    # Read RGB images (drop alpha channel)
    imgs = read_images_from_directory(os.path.join(sequence, str(idx)), prefix=None)[..., :3] # images are 1-indexed in the directory
    h, w = imgs.shape[1:3]

    # Read DUST3R depth files
    depth_dir = os.path.join(sequence, f'{duster_label}')
    depth_files = sorted(glob.glob(depth_dir + f'/*_scene.npz'))[:imgs.shape[0]]  # Ensure we only read as many depth files as there are images
    idx_depth = idx
    depth = []
    for depth_file in depth_files:
        depth_data = np.load(depth_file)
        depth.append(cv2.resize(
                depth_data['depths'][idx_depth], (w, h), interpolation=cv2.INTER_NEAREST
            ))
    depth = np.asarray(depth)/SCENE_SCALE
    # depth = np.clip(depth, None, 4.0)  # Clamp depth to a maximum of 4.0 meters

    # Optionally interpolate depth at query points
    if query_points is not None:
        q_depth = np.zeros((depth.shape[0], query_points.shape[1]), dtype=np.float32)
        for i in range(depth.shape[0]):
            x = query_points[:, 0]
            y = query_points[:, 1]
            x0 = x
            y0 = y
            x1 = x0 + 1
            y1 = y0 + 1

            x0 = torch.clamp(x0, 0, depth.shape[2] - 1)
            x1 = torch.clamp(x1, 0, depth.shape[2] - 1)
            y0 = torch.clamp(y0, 0, depth.shape[1] - 1)
            y1 = torch.clamp(y1, 0, depth.shape[1] - 1)

            Ia = depth[i, y0, x0]
            Ib = depth[i, y1, x0]
            Ic = depth[i, y0, x1]
            Id = depth[i, y1, x1]

            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            q_depth[i] = wa * Ia + wb * Ib + wc * Ic + wd * Id
        depth = q_depth
    
    # Normalize intrinsics
    intrinsics_normal = np.diag([1/w, 1/h, 1]) @ intrinsics @ np.diag([1, -1, -1])

    return depth, imgs, intrinsics_normal, intrinsics, extrinsics, cam_ID
