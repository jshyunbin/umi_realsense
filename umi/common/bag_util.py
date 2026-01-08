from bagpy import bagreader
from datetime import datetime
import pandas as pd


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