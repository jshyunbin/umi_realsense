"""
python scripts_slam_pipeline/01_extract_videos.py /data/UMI
"""
import sys
import os
import pathlib
import click
import multiprocessing
import concurrent.futures
from tqdm import tqdm


# --- Environment Setup (Boilerplate) ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.bag_util import (
    bag_get_fps,
    process_bag_to_mp4,
    BAG_VID_TOPIC,
    BAG_VID_ENC
)

# --- Main CLI Function ---

@click.command(help='Extracts MP4 video from raw_bag.bag files in demos subdirectories.')
@click.option('-f', '--fps', type=int, default=30, help="Target FPS for the output MP4 video.")
@click.option('-n', '--num_workers', type=int, default=None, help="Number of concurrent processes.")
@click.argument('session_dir', nargs=-1)
def main(fps, num_workers, session_dir):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        print(f"Using {num_workers} workers.")

    print("--- Starting RealSense BAG to MP4 Video Conversion ---")
    
    vid_names = ['depth', 'ir_l', 'ir_r', 'color']

    for session in session_dir:
        session_path = pathlib.Path(os.path.expanduser(session)).absolute()
        input_dir = session_path.joinpath('demos')
        
        # Find all raw_bag.bag paths
        input_bag_paths = [x for x in input_dir.glob('*/raw_bag.bag')]
        
        if not input_bag_paths:
            print(f"Warning: No 'raw_bag.bag' found in directories under {input_dir}. Skipping session.")
            continue

        print(f'Found {len(input_bag_paths)} BAG files for MP4 conversion.')

        total_tasks = len(input_bag_paths) * len(vid_names)
        
        done = set()

        # ProcessPoolExecutor를 사용하여 병렬 처리 (CV 작업은 CPU 바인딩이므로 프로세스 풀 사용)
        with tqdm(total=total_tasks, desc="Converting BAG to MP4") as pbar: 
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                
                for bag_path in input_bag_paths:
                    bag_dir = bag_path.parent 
                    
                    bag_fps = bag_get_fps(str(bag_path))
                    bag_fps = [round(f) for f in bag_fps]
                    if bag_fps != [fps]*4:
                        print(f"[WARNING] BAG file {bag_path.name} has non-matching FPS {bag_fps}, expected {fps}. Skipping.")
                        pbar.update(len(vid_names))
                        continue
                    
                    for vid_name in vid_names:
                        mp4_path = bag_dir.joinpath(f'{vid_name}_video.mp4')
                        
                        # Skip if MP4 already exists
                        if mp4_path.is_file():
                            print(f"[INFO] {bag_dir.name}/{vid_name}_video.mp4 already exists. Skipping.")
                            pbar.update()
                            continue

                        mp4_future = executor.submit(
                            process_bag_to_mp4, bag_path, mp4_path, BAG_VID_TOPIC[vid_name], fps, BAG_VID_ENC[vid_name])
                        futures.add(mp4_future)
                    
                        if len(futures) >= num_workers:
                            completed, futures = concurrent.futures.wait(futures,
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            done.update(completed)
                            pbar.update(len(completed))
                            

                while futures:
                    completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    done.update(completed)
                    pbar.update(len(completed))

        results = [x.result() for x in done if x.result() is True]
        errors = [x.result() for x in done if x.result() is False]
        
        print("\nDone! Summary:")
        print(f"  Total successful MP4/IMU conversions: {len(results)}")
        print(f"  Total conversion failures: {len(errors)}")

# %%
if __name__ == "__main__":
    main()