from bagpy import bagreader
from datetime import datetime
import pandas as pd
import rosbag
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import numpy as np


BAG_VID_NAME = [
    'depth',
    'ir_l',
    'ir_r',
    'color'
]

BAG_VID_ENC = {
    'depth': 'mono16', 
    'ir_l': '8UC1', 
    'ir_r': '8UC1', 
    'color': 'bgr8'
}

BAG_IMU_TOPIC = {
    'accel': '/device_0/sensor_2/Accel_0/imu/data',
    'gyro': '/device_0/sensor_2/Gyro_0/imu/data'
}

BAG_VID_TOPIC = {
    'depth': '/device_0/sensor_0/Depth_0/image/data', 
    'ir_l': '/device_0/sensor_0/Infrared_1/image/data', 
    'ir_r': '/device_0/sensor_0/Infrared_2/image/data',
    'color': '/device_0/sensor_1/Color_0/image/data'
}


def process_bag_to_csv(bag_path, csv_path, start_time=0.0):
    """
    Extract IMU data from BAG file and save as CSV.
    Args:
        bag_path (pathlib.Path): Path to the source raw_bag.bag file.
        csv_path (pathlib.Path): Path to the target imu_data.csv file.
        start_time (double): Start time in system time seconds 
    """
    b = bagreader(str(bag_path), verbose=False)
    data_csv = [b.message_by_topic(BAG_IMU_TOPIC[imu_type]) for imu_type in ['accel', 'gyro']]
    df = [pd.read_csv(dc) for dc in data_csv]
    df[1]['Time'] = df[1]['header.stamp.secs'] + df[1]['header.stamp.nsecs']*1e-9
    df[0]['Time'] = df[0]['header.stamp.secs'] + df[0]['header.stamp.nsecs']*1e-9
    df[1] = df[1][['Time', 'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z']].sort_values('Time')
    df[0] = df[0][['Time', 'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']].sort_values('Time')

    df[0]['angular_velocity.x'] = np.interp(df[0]['Time'], df[1]['Time'], df[1]['angular_velocity.x'])
    df[0]['angular_velocity.y'] = np.interp(df[0]['Time'], df[1]['Time'], df[1]['angular_velocity.y'])
    df[0]['angular_velocity.z'] = np.interp(df[0]['Time'], df[1]['Time'], df[1]['angular_velocity.z'])
    df = df[0]
    df.to_csv(csv_path, index=False)
    
    

def process_bag_to_mp4(bag_path, mp4_path, vid_name, fps=30):
    """
    Core conversion function: Reads the color image topic from a BAG file and saves it as an MP4.
    
    Args:
        bag_path (pathlib.Path): Path to the source raw_bag.bag file.
        mp4_path (pathlib.Path): Path to the target raw_video.mp4 file.
        vid_name (str): ROS topic name for the color image stream.
        fps (int): Target frame rate for the output MP4 video.
    
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    
    bridge = CvBridge()
    video_writer = None
    success = False
    timestamp_path = mp4_path.parent.joinpath('timestamps')
    timestamp_path.mkdir(exist_ok=True, parents=True)
    
    try:
        with rosbag.Bag(str(bag_path), 'r') as bag:
            with open(timestamp_path.joinpath(f'{vid_name}.txt'), 'w') as ts_f:
                is_first_frame = True
                
                # Read BAG file and write frames to video
                for _, msg, _ in bag.read_messages(topics=[BAG_VID_TOPIC[vid_name]]):
                    # Check for correct message type
                    if msg._type != Image._type: 
                        continue
                    ts_f.write(str(msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9) + '\n')
                    # Convert image message to OpenCV Mat object
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding=BAG_VID_ENC[vid_name])
                    
                    if is_first_frame:
                        # VideoWriter setup (Using 'mp4v' codec for broad compatibility)
                        if vid_name == 'color':
                            height, width, _ = cv_image.shape
                        else:
                            height, width = cv_image.shape
                            
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        
                        # Use absolute path for robustness
                        video_writer = cv2.VideoWriter(str(mp4_path.absolute()), fourcc, fps, (width, height), isColor=(vid_name == "color"))
                        is_first_frame = False

                    if video_writer is not None:
                        video_writer.write(cv_image)
            
                success = video_writer is not None and not is_first_frame
            
    except Exception as e:
        print(f"[MP4] Conversion failed for {bag_path.parent.name}/{mp4_path.name} due to an exception: {e}")
        success = False
    finally:
        if video_writer is not None:
            video_writer.release()
            
    if not success:
        print(f"[MP4] Conversion failed for {bag_path.parent.name}: No image frames found or initialization failed.")
        
    return success

def bag_get_start_datetime(file_path: str) -> datetime:
    """
    Reads bag file and returns system time of the first message.
    [Warning] This function does not get exact start time of the bag file
              use it only for estimating up to seconds accuracy.
    """
    try:
        reader = bagreader(file_path, verbose=False)
    except Exception as e:
        print(f"Error reading bag file {file_path}: {e}")
        return datetime.now()
    
    meta = reader.message_by_topic('/device_0/sensor_0/Depth_0/image/metadata')
    md = pd.read_csv(meta)
    return datetime.fromtimestamp(float(md.at[8, 'value'])/1000.0)
    
def bag_get_camera_serial(file_path: str) -> str:
    """
    Returns a camera serial number extracted from the BAG file.
    """
    try:
        reader = bagreader(file_path, verbose=False)
    except Exception as e:
        print(f"Error reading bag file {file_path}: {e}")
        return None
    
    meta = reader.message_by_topic('/device_0/info')
    md = pd.read_csv(meta)
    serial = md.at[1, 'value']
    return serial


def bag_get_fps(file_path: str) -> float:
    """
    Estimates the FPS of the color video stream in the BAG file.
    """
    try:
        reader = bagreader(file_path, verbose=False)
    except Exception as e:
        print(f"Error reading bag file {file_path}: {e}")
        return 30.0  # default FPS
    
    fps = reader.topic_table.loc[reader.topic_table['Types'] == 'sensor_msgs/Image', 'Frequency'].values
    return fps