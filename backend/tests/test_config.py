"""Test configuration constants and values."""

import pytest
from datetime import datetime
from config import OPTIMAL_REWARD_WEIGHTS, TRAINING_CONFIG, WEATHER_MAPPING

class TestConfig:
    """Test configuration constants and values."""
    
    def test_optimal_reward_weights_structure(self):
        """Test that optimal reward weights have expected structure."""
        expected_keys = {
            'step_penalty', 'dist_error_coef', 'move_cost_coef',
            'empty_station_penalty', 'full_station_penalty', 'trip_failure_penalty',
            'diversity_bonus', 'successful_trip_bonus', 'proactive_bonus',
            'distance_cost_factor'
        }
        
        assert set(OPTIMAL_REWARD_WEIGHTS.keys()) == expected_keys
        
        # Check that all values are numeric
        for key, value in OPTIMAL_REWARD_WEIGHTS.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"
    
    def test_training_config_structure(self):
        """Test training configuration structure."""
        assert 'optimal_training_period' in TRAINING_CONFIG
        assert 'test_period' in TRAINING_CONFIG
        assert 'default_stations' in TRAINING_CONFIG
        
        # Check date format
        start_date = TRAINING_CONFIG['optimal_training_period']['start']
        end_date = TRAINING_CONFIG['optimal_training_period']['end']
        
        # Should be valid dates
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    
    def test_weather_mapping(self):
        """Test weather condition mapping."""
        assert isinstance(WEATHER_MAPPING, dict)
        assert 'Clear' in WEATHER_MAPPING
        assert 'Sunny' in WEATHER_MAPPING
        
        # All values should be integers
        for condition, value in WEATHER_MAPPING.items():
            assert isinstance(value, int)