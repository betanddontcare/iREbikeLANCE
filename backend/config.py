"""
Configuration and constants for bike sharing RL application.
"""

import os

# Optimal reward weights from research paper (Table 5)
OPTIMAL_REWARD_WEIGHTS = {
    'step_penalty': -0.003,
    'dist_error_coef': -0.12,
    'move_cost_coef': -0.003,
    'empty_station_penalty': -0.25,
    'full_station_penalty': -0.25,
    'trip_failure_penalty': -1.5,
    'diversity_bonus': 0.03,
    'successful_trip_bonus': 0.02,
    'proactive_bonus': 0.01,
    'distance_cost_factor': -0.002
}

# Training configuration
TRAINING_CONFIG = {
    'optimal_training_period': {
        'start': '2023-05-15',
        'end': '2023-11-15'
    },
    'test_period': {
        'start': '2023-11-16', 
        'end': '2023-11-22'
    },
    'default_stations': 100,
    'default_timesteps': 100000,
    'time_step_minutes': 15,
    'max_episode_length': 96,
    'max_bikes_per_station': 50
}

# File paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
SNAPSHOTS_JSON_SUBDIR = 'snapshots_json_data'
TRAINED_MODELS_SUBDIR = 'trained_models'

# Weather condition mapping
WEATHER_MAPPING = {
    'Clear': 0, 'Sunny': 1, 'Partly cloudy': 2, 'Cloudy': 3, 'Overcast': 4,
    'Patchy rain possible': 5, 'Moderate rain at times': 6, 'Heavy rain at times': 7,
    'Moderate or heavy rain shower': 8
}

# Required data files
REQUIRED_FILES = {
    "trips": ('trips.json', 'Trip data'),
    "bike_counts": ('stations.json', 'Station bike counts'),
    "station_info": ('output_1.json', 'Station information'),
    "attributes": ('attributes.geojson', 'Station attributes'),
    "weather": ('weather.csv', 'Weather data')
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'threaded': True
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(processName)s - %(threadName)s - %(message)s',
    'log_file': 'app.log'
}