"""Test bike sharing environment."""

import pytest
import numpy as np
import pandas as pd
from environment import BikeSharingEnv

class TestEnvironment:
    """Test bike sharing environment."""
    
    @pytest.fixture
    def sample_env_data(self):
        """Sample data for environment testing."""
        timestamps = pd.date_range('2023-05-15 10:00:00', periods=4, freq='15min')
        
        return {
            'grouped_trips': {
                ts: [{'from_station': '101', 'to_station': '102'}] for ts in timestamps
            },
            'grouped_bike_counts': {
                ts: {'101': 10, '102': 5, '103': 8} for ts in timestamps
            },
            'station_uids': ['101', '102', '103'],
            'station_capacity': {'101': 20, '102': 25, '103': 15}
        }
    
    def test_environment_initialization(self, sample_env_data):
        """Test environment initialization."""
        env = BikeSharingEnv(
            grouped_trips=sample_env_data['grouped_trips'],
            grouped_bike_counts=sample_env_data['grouped_bike_counts'],
            station_uids=sample_env_data['station_uids'],
            station_capacity=sample_env_data['station_capacity']
        )
        
        assert env.num_stations == 3
        assert len(env.station_uids) == 3
        assert env.max_bikes == 50
        assert env.time_step_minutes == 15
    
    def test_environment_reset(self, sample_env_data):
        """Test environment reset functionality."""
        env = BikeSharingEnv(
            grouped_trips=sample_env_data['grouped_trips'],
            grouped_bike_counts=sample_env_data['grouped_bike_counts'],
            station_uids=sample_env_data['station_uids'],
            station_capacity=sample_env_data['station_capacity']
        )
        
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] > 0
        assert env.current_step == 0
        assert env.completed_trips == 0
        assert env.failed_trips == 0