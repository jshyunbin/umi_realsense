"""
python scripts_slam_pipeline/02_create_map.py -i /data/mapping 
"""

# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import numpy as np
import cv2
# from umi.common.cv_util import draw_predefined_mask
from umi.common.cv_util_realsense import (
    draw_rgb_predefined_mask,
    draw_im_l_infrared_mask,
    draw_im_r_infrared_mask,
    RGB_IMG_SHAPE,
    IR_IMG_SHAPE
)

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
def main(input_dir, map_path, no_mask):
    bag_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()

    for fn in ['raw_bag.bag', 'color_video.mp4', 'depth_video.mp4', 'ir_l_video.mp4', 'ir_r_video.mp4']:
        assert bag_dir.joinpath(fn).is_file()

    if map_path is None:
        map_path = bag_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = bag_dir.joinpath('mapping_camera_trajectory.csv')
    video_l_path = bag_dir.joinpath('ir_l_video.mp4')
    video_r_path = bag_dir.joinpath('ir_r_video.mp4')
    imu_csv_path = bag_dir.joinpath('imu_data.csv')

    if not no_mask:
        # left, right
        ir_l_slam_mask_path = bag_dir.joinpath('ir_l_slam_mask.png')
        ir_r_slam_mask_path = bag_dir.joinpath('ir_r_slam_mask.png')

        slam_mask = np.zeros(IR_IMG_SHAPE, dtype=np.uint8)
        slam_mask = draw_im_l_infrared_mask(slam_mask, color=255)
        cv2.imwrite(str(ir_l_slam_mask_path.absolute()), slam_mask)

        slam_mask = np.zeros(IR_IMG_SHAPE, dtype=np.uint8)
        slam_mask = draw_im_r_infrared_mask(slam_mask, color=255)
        cv2.imwrite(str(ir_r_slam_mask_path.absolute()), slam_mask)


    map_mount_source = pathlib.Path(map_path)

    ORB_SLAM3_ROOT = pathlib.Path("/data/ORB_SLAM3").expanduser()
    binary_path = ORB_SLAM3_ROOT.joinpath("Examples/Stereo/stereo_realsense_slam")
    setting_path = ORB_SLAM3_ROOT.joinpath("Examples/Stereo/RealSense_D435i.yaml")
    voca_path = ORB_SLAM3_ROOT.joinpath("Vocabulary/ORBvoc.txt")

    cmd = [
        str(binary_path),
        "--setting", str(setting_path), 
        "--vocabulary", str(voca_path),
        "--input_video_l", str(video_l_path),
        "--input_video_r", str(video_r_path),
        "--output_trajectory_csv", str(csv_path),
        "--save_map", str(map_mount_source),
    ]

    if not no_mask:
        cmd.extend([
            "--ir_l_mask", str(ir_l_slam_mask_path),
            "--ir_r_mask", str(ir_r_slam_mask_path),
        ])
    
    print(cmd)

    stdout_path = bag_dir.joinpath('slam_stdout.txt')
    stderr_path = bag_dir.joinpath('slam_stderr.txt')

    result = subprocess.run(
        cmd,
        cwd=str(bag_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print(f"[INFO] create map {result=}")


# %%
if __name__ == "__main__":
    main()
