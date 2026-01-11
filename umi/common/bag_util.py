from bagpy import bagreader
from datetime import datetime
import pandas as pd


BAG_VIDEO_ENC = {
    'depth': 'mono16', 
    'ir_l': '8UC1', 
    'ir_r': '8UC1', 
    'color': 'bgr8'
}

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