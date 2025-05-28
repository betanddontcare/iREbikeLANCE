"""
Utility functions for bike sharing RL application.
"""

import os
import sys
import math
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from config import DATA_DIR, LOGGING_CONFIG

def setup_logging():
    """Configure logging with Windows compatibility."""
    if sys.platform.startswith('win'):
        try:
            os.system('chcp 65001 > nul')
        except:
            pass
    
    class WindowsSafeFormatter(logging.Formatter):
        def format(self, record):
            msg = super().format(record)
            if sys.platform.startswith('win'):
                msg = msg.replace('✅', '[OK]')
                msg = msg.replace('⚠️', '[WARN]')
                msg = msg.replace('❌', '[ERROR]')
                msg = msg.replace('🎯', '[TARGET]')
                msg = msg.replace('📊', '[STATS]')
            return msg
    
    formatter = WindowsSafeFormatter(LOGGING_CONFIG['format'])
    
    file_handler = logging.FileHandler(LOGGING_CONFIG['log_file'], mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        handlers=[file_handler, console_handler],
        force=True
    )

def get_data_file_path(filename, sub_dir=None):
    """Get full path to data file."""
    base = DATA_DIR
    if sub_dir:
        base = os.path.join(DATA_DIR, sub_dir)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points in kilometers."""
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def build_distance_matrix(station_info_path, station_uids):
    """Build distance matrix between all station pairs."""
    with open(station_info_path, 'r', encoding='utf-8') as f:
        station_data = json.load(f)
    
    station_coords = {}
    for station in station_data.get('stations_data', []):
        uid = str(station.get('uid'))
        if uid in station_uids:
            if 'geoCoords' in station:
                lat = station['geoCoords'].get('lat')
                lng = station['geoCoords'].get('lng')
                if lat is not None and lng is not None:
                    station_coords[uid] = (float(lat), float(lng))
    
    n_stations = len(station_uids)
    distance_matrix = np.zeros((n_stations, n_stations), dtype=np.float32)
    
    for i, uid_i in enumerate(station_uids):
        if uid_i not in station_coords:
            continue
        lat_i, lon_i = station_coords[uid_i]
        
        for j, uid_j in enumerate(station_uids):
            if uid_j not in station_coords or j <= i:
                continue
            lat_j, lon_j = station_coords[uid_j]
            dist = haversine_distance(lat_i, lon_i, lat_j, lon_j)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix, station_coords

def handle_env_reset(env_reset_result):
    """Handle both old gym and new gymnasium reset() return values."""
    if isinstance(env_reset_result, tuple):
        return env_reset_result
    else:
        return env_reset_result, {}

def find_closest_timestamp(requested_timestamp, available_timestamps):
    """Find the closest available timestamp to the requested one."""
    try:
        requested_dt = pd.to_datetime(requested_timestamp)
        available_dts = [pd.to_datetime(ts) for ts in available_timestamps]
        
        time_diffs = [abs((dt - requested_dt).total_seconds()) for dt in available_dts]
        closest_idx = time_diffs.index(min(time_diffs))
        
        return available_timestamps[closest_idx]
    except Exception:
        return available_timestamps[0] if available_timestamps else None

def validate_date_range(start_date, end_date):
    """Validate date range format and logic."""
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_dt >= end_dt:
            return False, "Start date must be before end date"
        
        return True, None
    except ValueError as e:
        return False, f"Invalid date format: {str(e)}"

def convert_max_bikes_param(max_bikes_config):
    """Convert max bikes parameter to appropriate type."""
    if max_bikes_config == 'all' or max_bikes_config is None:
        return None
    try:
        return int(max_bikes_config)
    except (ValueError, TypeError):
        return None

def calculate_performance_category(success_rate):
    """Calculate performance category based on success rate."""
    if success_rate >= 80:
        return 'Excellent', 'green'
    elif success_rate >= 65:
        return 'Good', 'blue'
    elif success_rate >= 50:
        return 'Fair', 'yellow'
    else:
        return 'Needs Improvement', 'red'

def get_performance_recommendations(success_rate):
    """Get performance improvement recommendations."""
    if success_rate < 50:
        return [
            'Consider increasing trip failure penalty (more negative)',
            'Reduce step penalty to allow more exploration',
            'Try training for more timesteps'
        ]
    elif success_rate < 70:
        return [
            'Fine-tune reward weights for better balance',
            'Consider increasing diversity bonus'
        ]
    else:
        return [
            'Excellent performance! Consider testing on longer periods',
            'Model ready for deployment'
        ]