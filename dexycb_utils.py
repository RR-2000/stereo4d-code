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

def read_images_from_directory(dir :str, prefix = None):
    """
    Reads all images from a directory, optionally filtering by prefix.
    Supports PNG, JPG, JPEG, and TIFF formats.
    Returns a numpy array of stacked images.
    """
    print(f"Reading images from directory: {dir} with prefix: {prefix}")
    images = []
    if prefix is None:
        for filename in sorted(os.listdir(dir)):
            filepath = os.path.join(dir, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = np.array(Image.open(filepath))
                images.append(img)
            elif os.path.isfile(filepath) and filename.lower().endswith(('.tiff')):
                # Read TIFF images as float64 and take the first channel
                img = np.asarray(imageio.v2.imread(pathlib.Path(filepath).read_bytes(), format="tiff"), dtype=np.float64)[:,:, 0]
                images.append(img)
    else:
        for filename in sorted(os.listdir(dir)):
            if filename.startswith(prefix):
                filepath = os.path.join(dir, filename)
                if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = np.array(Image.open(filepath))
                    images.append(img)
                elif os.path.isfile(filepath) and filename.lower().endswith(('.tiff')):
                    img = np.asarray(imageio.v2.imread(pathlib.Path(filepath).read_bytes(), format="tiff"), dtype=np.float64)[:,:, 0]
                    images.append(img)

    return np.stack(images, axis=0)

def read_cam_dex(sequence :str, idx :int, query_points = None, depth: str = 'gt', num_cams: int = 8):
    """
    Reads camera and depth data for a given sequence and view index.
    Selects the appropriate depth type: 'gt', 'vggt', or 'dust3r'.
    """
    if depth == 'gt':
        return read_cam_dex_gt(sequence, idx, query_points)
    elif depth == 'vggt':
        return read_cam_dex_vggt(sequence, idx, query_points, num_cams=num_cams)
    elif depth == 'dust3r':
        return read_cam_dex_dust3r(sequence, idx, query_points, num_cams=num_cams)
    else:
        raise ValueError(f"Unknown depth type: {depth}. Use 'gt', 'vggt', or 'dust3r'.")

def read_cam_dex_gt(sequence :str, idx :int, query_points = None):
    """
    Reads ground truth camera intrinsics, extrinsics, depth, and RGB images for a given view.
    """
    meta = np.load(os.path.join(sequence, f'view_{idx:02d}', 'intrinsics_extrinsics.npz'))

    cam_ID = idx
    intrinsics = meta['intrinsics'][:3, :3]
    extrinsics = meta['extrinsics']
    # Read depth and RGB images
    depth = read_images_from_directory(os.path.join(sequence, f'view_{idx:02d}', 'depth'))/1000

    # # Clamp depth at 90th percentile
    # percentile_90 = np.percentile(depth, 90)
    # print(f"90th percentile depth: {percentile_90:.2f} m")
    depth = np.clip(depth, None, 4.0)  # Clamp depth to a maximum of 4.0 meters

    imgs = read_images_from_directory(os.path.join(sequence, f'view_{idx:02d}', 'rgb'))
    w, h = imgs.shape[1:3]

    # Normalize intrinsics
    intrinsics_normal = np.diag([1/w, 1/h, 1]) @ intrinsics @ np.diag([1, -1, -1])

    return depth, imgs, intrinsics_normal, intrinsics, extrinsics, cam_ID

def read_cam_dex_vggt(sequence :str, idx :int, depth_euclid = False, query_points = None, num_cams = 8):
    """
    Reads VGG-trained depth, camera intrinsics, extrinsics, and images for a given view.
    Optionally interpolates depth at query points.
    """
    meta = np.load(os.path.join(sequence, f'view_{idx:02d}', 'intrinsics_extrinsics.npz'))

    cam_ID = idx
    intrinsics = meta['intrinsics'][:3, :3]
    extrinsics = meta['extrinsics']

    # Select label based on number of cameras
    vggt_label = ''
    if num_cams == 8:
        vggt_label = 'vggt_dex_views_01234567_v2'
    elif num_cams == 6:
        vggt_label = 'vggt_dex_views_013467_v2'
    elif num_cams == 4:
        vggt_label = 'vggt_dex_views_0123_v2'

    cam_ID = idx
    vggt_path = os.path.join(sequence, f'{vggt_label}/subject-01',  f'view_{idx:02d}')
    num_frame = len(glob.glob(vggt_path + '/*.npz'))
    depth = []
    for frame_id in range(num_frame):
        depth.append(np.load(os.path.join(vggt_path, 'depth', f'{frame_id:05d}.npy')))


    depth = np.asarray(depth)
    depth = np.clip(depth, None, 4.0)  # Clamp depth to a maximum of 4.0 meters

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

    # Read RGB images (drop alpha channel)
    imgs = read_images_from_directory(os.path.join(sequence, f'view_{idx:02d}', 'rgb'), prefix=None)[..., :3]
    
    w, h = imgs.shape[1:3]
    # Normalize intrinsics
    intrinsics_normal = np.diag([1/w, 1/h, 1]) @ intrinsics @ np.diag([1, -1, -1])

    return depth, imgs, intrinsics_normal, intrinsics, extrinsics, cam_ID

def read_cam_dex_dust3r(sequence :str, idx :int, depth_euclid = False, query_points = None, num_cams = 8):
    """
    Reads DUST3R-predicted depth, camera intrinsics, extrinsics, and images for a given view.
    Optionally interpolates depth at query points.
    """
    meta = np.load(os.path.join(sequence, f'view_{idx:02d}', 'intrinsics_extrinsics.npz'))

    cam_ID = idx
    intrinsics = meta['intrinsics'][:3, :3]
    extrinsics = meta['extrinsics']

    # Select label based on number of cameras
    duster_label = ''
    if num_cams == 8:
        duster_label = 'duster-views-01234567'
    elif num_cams == 6:
        duster_label = 'duster-views-013467'
    elif num_cams == 4:
        duster_label = 'duster-views-0123'


    # Read RGB images (drop alpha channel)
    imgs = read_images_from_directory(os.path.join(sequence, f'view_{idx:02d}', 'rgb'), prefix=None)[..., :3]
    w, h = imgs.shape[1:3]

    # Read DUST3R depth files
    depth_dir = os.path.join(sequence, f'{duster_label}')
    depth_files = sorted(glob.glob(depth_dir + f'/*_scene.npz'))
    depth = []
    for depth_file in depth_files:
        depth_data = np.load(depth_file)
        depth.append(cv2.resize(
                depth_data['depths'][idx], (h, w), interpolation=cv2.INTER_NEAREST
            ))
    depth = np.asarray(depth)
    depth = np.clip(depth, None, 4.0)  # Clamp depth to a maximum of 4.0 meters

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
