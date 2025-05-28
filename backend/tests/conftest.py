"""Shared test fixtures and configuration."""

import pytest
import numpy as np
import pandas as pd
import json
import os
import tempfile
import shutil
from unittest.mock import Mock

@pytest.fixture(scope="session")
def sample_reward_weights():
    """Sample reward weights for testing."""
    return {
        'step_penalty': -0.01,
        'dist_error_coef': -0.1,
        'move_cost_coef': -0.005,
        'empty_station_penalty': -0.2,
        'full_station_penalty': -0.2,
        'trip_failure_penalty': -1.0,
        'diversity_bonus': 0.05,
        'successful_trip_bonus': 0.01,
        'proactive_bonus': 0.02,
        'distance_cost_factor': -0.001
    }

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_bike_counts_data():
    """Sample bike counts data."""
    return {
        "2023-05-15 10:00:00": {
            "101": 5,
            "102": 8,
            "103": 3
        },
        "2023-05-15 10:15:00": {
            "101": 4,
            "102": 9,
            "103": 2
        }
    }

@pytest.fixture
def sample_station_info_data():
    """Sample station info data."""
    return {
        "stations_data": [
            {
                "uid": 101,
                "availabilityStatus": {"bikeRacks": 20},
                "geoCoords": {"lat": 52.2297, "lng": 21.0122}
            },
            {
                "uid": 102,
                "availabilityStatus": {"bikeRacks": 25},
                "geoCoords": {"lat": 52.2397, "lng": 21.0222}
            },
            {
                "uid": 103,
                "availabilityStatus": {"bikeRacks": 15},
                "geoCoords": {"lat": 52.2197, "lng": 21.0022}
            }
        ]
    }