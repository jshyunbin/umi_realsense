"""
python scripts_slam_pipeline/02_create_map.py -i /data/test_collection/demos/mapping
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
from tqdm import tqdm
import numpy as np
import cv2
from umi.common.cv_util_realsense import (
    draw_im_l_infrared_mask,
    draw_im_r_infrared_mask,
    IR_IMG_SHAPE
)

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-s', '--stereo', is_flag=True, default=False, help='Use stereo SLAM instead of stereo-inertial SLAM')
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
def main(input_dir, map_path, stereo, no_mask):
    if stereo:
        print("[WARN] Not using IMU data for map creation.")
    
    bag_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    extract_dir = bag_dir.joinpath('extracted_data')

    for fn in ['color_video.mp4', 'depth_video.mp4', 'ir_l_video.mp4', 'ir_r_video.mp4']:
        assert extract_dir.joinpath(fn).is_file()

    if map_path is None:
        map_path = bag_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = bag_dir.joinpath('mapping_camera_trajectory.csv')
    video_l_path = extract_dir.joinpath('ir_l_video.mp4')
    video_r_path = extract_dir.joinpath('ir_r_video.mp4')
    df_csv_path = extract_dir.joinpath('timestamps.csv')

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
    if stereo:
        binary_path = ORB_SLAM3_ROOT.joinpath("Examples/Stereo/stereo_realsense_slam")
        setting_path = ORB_SLAM3_ROOT.joinpath("Examples/Stereo/RealSense_D435i.yaml")
        voca_path = ORB_SLAM3_ROOT.joinpath("Vocabulary/ORBvoc.txt")
    else:
        binary_path = ORB_SLAM3_ROOT.joinpath("Examples/Stereo-Inertial/realsense_slam")
        setting_path = ORB_SLAM3_ROOT.joinpath("Examples/Stereo-Inertial/RealSense_D435i.yaml")
        voca_path = ORB_SLAM3_ROOT.joinpath("Vocabulary/ORBvoc.txt")

    cmd = [
        str(binary_path),
        "--setting", str(setting_path), 
        "--vocabulary", str(voca_path),
        "--input_video_l", str(video_l_path),
        "--input_video_r", str(video_r_path),
        "--input_df_csv", str(df_csv_path),
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
    print(f"[INFO] create map {result.returncode=}")


# %%
if __name__ == "__main__":
    main()
