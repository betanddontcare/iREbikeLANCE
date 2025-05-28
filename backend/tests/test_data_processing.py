"""Test data processing functions."""

import pytest
import json
import os
from data_processing import (
    load_station_ids_from_bikes_file, 
    build_station_capacity_mapping,
    normalize_station_attributes
)

class TestDataProcessing:
    """Test data processing functions."""
    
    def test_load_station_ids_from_bikes_file(self, temp_dir, sample_bike_counts_data):
        """Test loading station IDs from bike counts file."""
        bike_counts_path = os.path.join(temp_dir, 'bike_counts.json')
        with open(bike_counts_path, 'w') as f:
            json.dump(sample_bike_counts_data, f)
        
        station_ids = load_station_ids_from_bikes_file(bike_counts_path)
        expected_ids = {'101', '102', '103'}
        assert station_ids == expected_ids
    
    def test_build_station_capacity_mapping(self, temp_dir, sample_bike_counts_data, sample_station_info_data):
        """Test building station capacity mapping."""
        bike_counts_path = os.path.join(temp_dir, 'bike_counts.json')
        station_info_path = os.path.join(temp_dir, 'station_info.json')
        
        with open(bike_counts_path, 'w') as f:
            json.dump(sample_bike_counts_data, f)
        
        with open(station_info_path, 'w') as f:
            json.dump(sample_station_info_data, f)
        
        capacity_map = build_station_capacity_mapping(bike_counts_path, station_info_path)
        
        assert capacity_map['101'] == 20.0
        assert capacity_map['102'] == 25.0
        assert capacity_map['103'] == 15.0
    
    def test_normalize_station_attributes(self):
        """Test station attributes normalization."""
        station_attributes = {
            '101': {'distance': 100, 'population': 1000},
            '102': {'distance': 200, 'population': 2000},
            '103': {'distance': 300, 'population': 3000}
        }
        
        normalized = normalize_station_attributes(station_attributes)
        
        # Check normalization (values should be between 0 and 1)
        for station_id, attrs in normalized.items():
            for attr_name, value in attrs.items():
                assert 0 <= value <= 1