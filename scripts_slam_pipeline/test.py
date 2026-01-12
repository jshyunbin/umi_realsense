"""
python ./scripts_slam_pipeline/test.py
"""
import os
import sys

import numpy as np
import cv2
import av
import pickle
import zarr


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.bag_util import bag_get_fps
from umi.common.cv_util_realsense import (
    draw_im_l_infrared_mask,
    draw_im_r_infrared_mask,
    draw_rgb_predefined_mask,
    inpaint_tag,
    get_image_transform,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer

if __name__ == "__main__":
    bag_path = "/data/UMI/demos/mapping/raw_bag.bag"
    vid_path = "/data/UMI/demos/mapping/color_video.mp4"
    dataset_path = "/data/UMI/dataset.zarr.zip"
    
    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
    
    print(replay_buffer.root.tree())
    
    
    # tag_detection_pkl_path = "/data/UMI/demos/mapping/tag_detection.pkl"
    # with open(tag_detection_pkl_path, 'rb') as f:
    #     tag_detection_results = pickle.load(f)
    
    
    # with av.open(str(vid_path)) as container:
        
    #     in_stream = container.streams.video[0]
        
    #     resize_tf = get_image_transform(
    #         in_res=(1920, 1080),
    #         out_res=(224, 224),
    #         # grayscale=True
    #     )

    #     for frame_idx, frame in enumerate(container.decode(in_stream)):
    #         if frame_idx == 0:
    #             print(f"Frame {frame_idx}: {frame.width}x{frame.height}, format={frame.format.name}")
                
    #             img = frame.to_ndarray(format='rgb24')
                
    #             # inpaint tags
    #             # this_det = tag_detection_results[frame_idx]
    #             # all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
    #             # for corners in all_corners:
    #             #     img = inpaint_tag(img, corners)
                
    #             # img = draw_im_l_infrared_mask(img, color=0, 
    #             #             mirror=False, gripper=True, finger=True)
                
    #             print(f"img shape: {img.shape}, dtype={img.dtype}")
    #             img = resize_tf(img)
    #             print(f"img shape: {img.shape}, dtype={img.dtype}")
    #             while(1):
    #                 cv2.imshow('frame', img)
    #                 k= cv2.waitKey(1) & 0xFF
    #                 if k == 27:
    #                     break