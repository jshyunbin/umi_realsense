from bagpy import bagreader
from datetime import datetime
import pandas as pd
import rosbag
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image


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


def process_bag_to_csv(bag_path, csv_path, imu_type):
    """
    Extract IMU data from BAG file and save as CSV.
    Args:
        bag_path (pathlib.Path): Path to the source raw_bag.bag file.
        csv_path (pathlib.Path): Path to the target imu_data.csv file.
        imu_topic_name (str): ROS topic name for the IMU data stream.
    """
    b = bagreader(str(bag_path))
    data_csv = b.message_by_topic(BAG_IMU_TOPIC[imu_type])
    df = pd.read_csv(data_csv)
    if imu_type == 'gyro':
        df = df[['Time', 'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z']]
    else:  # 'accel'
        df = df[['Time', 'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']]
    
    df.to_csv(csv_path, index=False)
    
    

def process_bag_to_mp4(bag_path, mp4_path, color_topic_name, fps=30, enc="bgr8"):
    """
    Core conversion function: Reads the color image topic from a BAG file and saves it as an MP4.
    
    Args:
        bag_path (pathlib.Path): Path to the source raw_bag.bag file.
        mp4_path (pathlib.Path): Path to the target raw_video.mp4 file.
        color_topic_name (str): ROS topic name for the color image stream.
        fps (int): Target frame rate for the output MP4 video.
    
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    # print(f"[MP4] Converting {bag_path.parent.name} to {mp4_path.name}...")
    
    bridge = CvBridge()
    video_writer = None
    success = False
    
    try:
        with rosbag.Bag(str(bag_path), 'r') as bag:
            is_first_frame = True
            
            # Read BAG file and write frames to video
            for _, msg, _ in bag.read_messages(topics=[color_topic_name]):
                # Check for correct message type
                if msg._type != Image._type: 
                     continue
                
                # Convert image message to OpenCV Mat object
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding=enc)
                
                if is_first_frame:
                    if enc == "bgr8":
                        height, width, _ = cv_image.shape
                    else:
                        height, width = cv_image.shape
                    
                    # VideoWriter setup (Using 'mp4v' codec for broad compatibility)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    # Use absolute path for robustness
                    video_writer = cv2.VideoWriter(str(mp4_path.absolute()), fourcc, fps, (width, height), isColor=(enc == "bgr8"))
                    is_first_frame = False
                    # print(f"[MP4] Writer initialized for {bag_path.parent.name}: {width}x{height} @ {fps} FPS")

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
    Returns the file's modification time (mtime) as a proxy for the start date/time.
    """
    try:
        reader = bagreader(file_path)
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
        reader = bagreader(file_path)
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
        reader = bagreader(file_path)
    except Exception as e:
        print(f"Error reading bag file {file_path}: {e}")
        return 30.0  # default FPS
    
    fps = reader.topic_table.loc[reader.topic_table['Types'] == 'sensor_msgs/Image', 'Frequency'].values
    return fps