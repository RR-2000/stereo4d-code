import os
import argparse
from io import BytesIO
import tarfile
from six.moves import urllib
import mediapy as media
import sys
sys.path.append('./')
from FlDot_utils import *
import os.path as osp
import numpy as np
from PIL import Image
import tqdm
import math
import cv2
import utils
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    input_height, input_width = self.INPUT_SIZE, self.INPUT_SIZE
    resize_ratio = min(1.0 * input_height / height, 1.0 * input_width / width)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.LANCZOS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]},
    )
    seg_map = batch_seg_map[0]
    seg_map = cv2.resize(seg_map, (width, height), interpolation=cv2.INTER_NEAREST)
    return seg_map



class SemanticSegmentor:
  def __init__(self):

    download_path = os.path.join('./deeplab', 'deeplab_model.tar.gz')
    if not osp.exists(download_path):
      os.makedirs('./deeplab', exist_ok=True)
      print('downloading model, this might take a while...')
      model_url = 'http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'
      urllib.request.urlretrieve(model_url, download_path)
      print('download completed! loading DeepLab model...')
    self.model = DeepLabModel(download_path)
    self.num_class_map = {
        'pascal': 21,
        'cityscapes': 19,
        'ade20k': 151,
    }

    self.label_names_map = {
        'pascal': [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
        ],
        'cityscapes': [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle'
        ],
        'ade20k': [
            'background', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling',
            'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person',
            'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair',
            'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror',
            'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
            'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
            'signboard', 'chest', 'counter', 'sand', 'sink', 'skyscraper',
            'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway',
            'case', 'pool', 'pillow', 'screen', 'stairway', 'river', 'bridge',
            'bookcase', 'blind', 'coffee', 'toilet', 'flower', 'book', 'hill',
            'bench', 'countertop', 'stove', 'palm', 'kitchen', 'computer', 'swivel',
            'boat', 'bar', 'arcade', 'hovel', 'bus', 'towel', 'light', 'truck',
            'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television',
            'airplane', 'dirt', 'apparel', 'pole', 'land', 'bannister', 'escalator',
            'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship',
            'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming',
            'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike',
            'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade', 'microwave',
            'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen2', 'blanket',
            'sculpture', 'hood', 'sconce', 'vase', 'traffic', 'tray', 'ashcan',
            'fan', 'pier', 'crt', 'plate', 'monitor', 'bulletin', 'shower',
            'radiator', 'glass', 'clock', 'flag'
        ],
    }
    # Get static ids
    self.static_id = []
    for class_name in ['building', 'road', 'earth', 'sidewalk', 'wall']:
      assert class_name in self.label_names_map['ade20k'], f'Class {class_name} not found in LABEL_NAMES_MAP["ade20k"]'
      self.static_id.append(self.label_names_map['ade20k'].index(class_name))
      
    # Get sky ids
    self.sky_id = []
    for class_name in ['sky']:
      assert class_name in self.label_names_map['ade20k'], f'Class {class_name} not found in LABEL_NAMES_MAP["ade20k"]'
      self.sky_id.append(self.label_names_map['ade20k'].index(class_name))
    
  
  def get_static_mask(self, label: np.ndarray)->np.ndarray:
    """
    Given a segmentation map, return a mask which indicates which pixel is static
    """
    static_mask = np.zeros_like(label)
    for i in range(len(self.static_id)):
      static_mask[label == self.static_id[i]] = 1
    return static_mask.astype(bool)
  
  def get_sky_mask(self, label: np.ndarray)->np.ndarray:
    """
    Given a segmentation map, return a mask which indicates which pixel is sky
    """
    sky_mask = np.zeros_like(label)
    for i in range(len(self.sky_id)):
      sky_mask[label == self.sky_id[i]] = 1
    return sky_mask.astype(bool)

  def get_static_track_mask(self, seg_maps: np.ndarray, track2d_xy: np.ndarray)->np.ndarray:
    """
    seg_maps: numpy array of shape (nframe, 512, 512), segmentation maps
    track2d_xy: numpy array of shape (npt, nframe, 2), track2d xy coordinates
    return: numpy array of shape (npt, nframe), True if the track xy location is in static, False otherwise
    """
    static_masks = []
    for fid in range(len(seg_maps)):
      chunk_size = 32000  # Set this to a manageable size
      remapped_chunks = []
      for start in range(0, track2d_xy.shape[0], chunk_size):
        end = min(start + chunk_size, track2d_xy.shape[0])
        chunk = track2d_xy[None, start:end, fid]
        remapped_chunk = cv2.remap(
            self.get_static_mask(seg_maps[fid])[..., None].astype(np.float32),
            np.clip(chunk, 0, 511).astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        remapped_chunks.append(remapped_chunk)
      remapped_chunks = np.concatenate(remapped_chunks, axis=0)
      static_masks.append(remapped_chunks)
    static_masks = np.stack(static_masks, axis=1).astype(bool)
    return static_masks
  
  def get_sky_track_mask(self, seg_maps: np.ndarray, track2d_xy: np.ndarray)->np.ndarray:
    """
    seg_maps: numpy array of shape (nframe, 512, 512), segmentation maps
    track2d_xy: numpy array of shape (npt, nframe, 2), track2d xy coordinates
    return: numpy array of shape (npt, nframe), True if the track xy location is in sky, False otherwise
    """
    sky_masks = []
    for fid in range(len(seg_maps)):
      chunk_size = 32000  # Set this to a manageable size
      remapped_chunks = []
      for start in range(0, track2d_xy.shape[0], chunk_size):
        end = min(start + chunk_size, track2d_xy.shape[0])
        chunk = track2d_xy[None, start:end, fid]
        remapped_chunk = cv2.remap(
            self.get_sky_mask(seg_maps[fid])[..., None].astype(np.float32),
            np.clip(chunk, 0, 511).astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        remapped_chunks.append(remapped_chunk)
      remapped_chunks = np.concatenate(remapped_chunks, axis=0)
      sky_masks.append(remapped_chunks)
    sky_masks = np.stack(sky_masks, axis=1).astype(bool)
    return sky_masks


def load_rgbd_cam(vid: str, root_dir: str, npz_folder: str,  hfov: float, new_imw=1, new_imh=1, depth: str = 'gt', num_cams: int = 8):
  """load rgb, depth, and camera"""
  input_dict = {'left': {'camera': [], 'depth': [], 'video': []}}
  # Load camera
  depths, rgbs, intrinsics_normal, intrinsics, extrinsics, cam_ID = read_cam_dex(root_dir, int(vid.split('-')[-1]), query_points = None, depth = depth, num_cams=num_cams)
  nfr = len(depths)
  input_dict['nfr'] = nfr
  for fid in range(nfr):
    intr_normalized = {
        'fx': intrinsics_normal[0,0] if len(intrinsics_normal.shape) == 2 else intrinsics_normal[fid][0, 0],
        'fy': -1*intrinsics_normal[1,1] if len(intrinsics_normal.shape) == 2 else -1*intrinsics_normal[fid][1, 1],
        'cx': -1*intrinsics_normal[0,2] if len(intrinsics_normal.shape) == 2 else -1*intrinsics_normal[fid][0, 2],
        'cy': -1*intrinsics_normal[1,2] if len(intrinsics_normal.shape) == 2 else -1*intrinsics_normal[fid][1, 2],
        'k1': 0,
        'k2': 0,
    }
    input_dict['left']['camera'].append(
        utils.CameraAZ(
            from_json={
                'extr': extrinsics[:3, :] if len(extrinsics.shape) == 2 else extrinsics[fid][:3, :],
                'intr_normalized': intr_normalized,
            }
        )
    )
  input_dict['left']['video'] = rgbs
  input_dict['left']['depth'] = depths
  return input_dict


def segmentation_main(vid: int, save_root: str, npz_folder: str, hfov: float, depth: str = 'gt', num_cams: int = 8):
  semantic_segmentor = SemanticSegmentor()
  video = read_images_from_directory(os.path.join(save_root, f'images', 'scaled_view'), idx = vid+1, prefix=None)[..., :3]
  vid = f'view-{vid:02d}'
  seg_maps = []
  for fid in tqdm.tqdm(range(len(video))):
    original_im = Image.fromarray(video[fid])
    seg_map = semantic_segmentor.model.run(original_im)
    seg_maps.append(seg_map)
  seg_maps = np.stack(seg_maps, axis=0)
  
  with open(
      osp.join(save_root, vid, vid + '-tapir_2d.pkl'), 'rb'
  ) as f:
    track2d = pickle.load(f)

  input_dict = load_rgbd_cam(vid, save_root, npz_folder, hfov, depth=depth, num_cams=num_cams)

  track3d = utils.Track3d(
      track2d['tracks'],
      track2d['visibles'],
      input_dict['left']['depth'],
      input_dict['left']['camera'],
      input_dict['left']['video'],
      track2d['query_points'],
  )
  # Discard sky points
  sky_masks = semantic_segmentor.get_sky_track_mask(
      seg_maps, track2d['tracks']
  )
  sky_masks = (sky_masks & (track2d['visibles'])).any(axis=1)
  print(f"percentage of sky points: {sky_masks.mean()}")

  # Get static tracks
  static_masks = semantic_segmentor.get_static_track_mask(
      seg_maps, track2d['tracks']
  )
  print(f"percentage of static points: {static_masks.mean()}")

  # Detect boundary points (where mask changed)
  mean_visible = (static_masks * track2d['visibles']).sum(axis=1) / track2d['visibles'].sum(axis=1)
  boundary_masks = (mean_visible > 0) & (mean_visible < 1)
  print(f"percentage of boundary points: {boundary_masks.mean()}")
  
  # Detect static points that drifts
  print("track3d shape:", track3d.track3d.shape)
  print("track3d visible shape:", track3d.visible_list.shape)
  displacement = utils.get_scene_motion_2d_displacement(track3d)
  drift_masks = ((displacement > 5) & track3d.visible_list & static_masks.astype(bool)).any(axis=1)
  print(f"percentage of drifting points: {drift_masks.mean()}")
  filtered_remaining_mask = ~(sky_masks | boundary_masks | drift_masks)
  print(f"remaining tracks percentage: {filtered_remaining_mask.mean()}")
  
  track3d = track3d.get_new_track(filtered_remaining_mask)
  displacement = displacement[filtered_remaining_mask]
  np.random.seed(2024)
  if filtered_remaining_mask.sum() > 64*64:
    samples = np.random.choice(
        np.arange(filtered_remaining_mask.sum()), 64 * 64, replace=False
    )
    track3d = track3d.get_new_track(samples)
  
  video3d_viz = utils.plot_3d_tracks_plt(
      input_dict['left']['video'],
      track3d,
  )
  media.write_video(
      osp.join(save_root, vid, vid + '-tapir_3dtrack_filtered.mp4'), video3d_viz, fps=30
  )
  track2d_filtered = {
      'tracks': track2d['tracks'][filtered_remaining_mask],
      'visibles': track2d['visibles'][filtered_remaining_mask],
      'query_points': track2d['query_points'][filtered_remaining_mask],
  }
  with open(
      osp.join(save_root, vid, vid + '-tapir_remove_drift_tracks.pkl'), 'wb'
  ) as f:
    pickle.dump(track2d_filtered, f)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--hfov', help='horizontal fov', type=float, default=1.09375)

  parser.add_argument('--num_views', help='number of views', type=int, default=5)
  parser.add_argument('--dir', help='path to views', type=str, default='/project/Thesis/data/FlDot/C01')
  parser.add_argument('--depth', help='Type of depth to use', type=str, choices=['dust3r'], default='dust3r')

  args = parser.parse_args()

  views = list(range(args.num_views))

  for idx in views:
    segmentation_main(idx, args.dir, args.dir, args.hfov, args.depth, args.num_views)


if __name__ == '__main__':
  main()