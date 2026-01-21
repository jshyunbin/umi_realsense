"""
python scripts_slam_pipeline/01_extract_videos.py /data/test_collection
"""
import sys
import os
import pathlib
import click
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import shutil


# --- Environment Setup (Boilerplate) ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.bag_util import (
    bag_get_fps,
    process_bag_to_mp4,
    process_bag_to_csv,
    BAG_VID_NAME,
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

    for session in session_dir:
        session_path = pathlib.Path(os.path.expanduser(session)).absolute()
        input_dir = session_path.joinpath('demos')
        
        # Find all raw_bag.bag paths
        input_bag_paths = [x for x in input_dir.glob('*/raw_bag.bag')]
        
        if not input_bag_paths:
            print(f"Warning: No 'raw_bag.bag' found in directories under {input_dir}. Skipping session.")
            continue

        print(f'Found {len(input_bag_paths)} BAG files for MP4 conversion.')

        total_tasks = len(input_bag_paths) * (len(BAG_VID_NAME) + 1)
        
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
                        pbar.update(len(BAG_VID_NAME) + 1)
                        continue
                    
                    for vid_name in BAG_VID_NAME:
                        mp4_path = bag_dir.joinpath(f'{vid_name}_video.mp4')
                        
                        # Skip if MP4 already exists
                        if mp4_path.is_file():
                            print(f"[INFO] {bag_dir.name}/{vid_name}_video.mp4 already exists. Skipping.")
                            pbar.update()
                            continue

                        mp4_future = executor.submit(
                            process_bag_to_mp4, bag_path, mp4_path, vid_name, fps)
                        futures.add(mp4_future)
                    
                        if len(futures) >= num_workers:
                            completed, futures = concurrent.futures.wait(futures,
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            done.update(completed)
                            pbar.update(len(completed))
                    
                    csv_path = bag_dir.joinpath(f'imu_data.csv')
                    
                    if csv_path.is_file():
                        print(f"[INFO] {bag_dir.name}/imu_data.csv already exists. Skipping.")
                        pbar.update()
                        continue

                    imu_future = executor.submit(
                        process_bag_to_csv, bag_path, csv_path)
                    futures.add(imu_future)

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
        
        print("Now matching timestamps...")
        
        with tqdm(total=len(input_bag_paths), desc="Matching Timestamps") as pbar:
            for bag_path in input_bag_paths:
                bag_dir = bag_path.parent
                df = pd.read_csv(bag_dir.joinpath('imu_data.csv'))
                
                for vid_name in BAG_VID_NAME:
                    ts_path = bag_dir.joinpath('timestamps', f'{vid_name}.txt')
                    if not ts_path.is_file():
                        print(f"[WARNING] Timestamp file {ts_path} not found. Skipping.")
                        pbar.update()
                        continue
                    with open(ts_path, 'r') as ts_f:
                        data = ts_f.read()
                        stamp_dict = dict()
                        lines = data.splitlines()
                        lines = [float(x) for x in lines]
                        
                        stamp_dict = {"Time": lines, f"{vid_name}_idx": list(range(len(lines)))}
                        
                        df = df.merge(pd.DataFrame.from_dict(stamp_dict), how='outer').sort_values('Time')
                shutil.rmtree(bag_dir.joinpath('timestamps'))
                os.remove(bag_dir.joinpath('imu_data.csv'))
                
                df["Time"] = df["Time"] - df["Time"].iloc[0]
                df.to_csv(bag_dir.joinpath('timestamps.csv'), index=False)
                pbar.update()

# %%
if __name__ == "__main__":
    main()