"""Test utility functions."""

import pytest
from utils import (
    haversine_distance, find_closest_timestamp, validate_date_range,
    convert_max_bikes_param, calculate_performance_category
)

class TestUtils:
    """Test utility functions."""
    
    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        # Test same point
        dist = haversine_distance(52.2297, 21.0122, 52.2297, 21.0122)
        assert dist == 0.0
        
        # Test known distance (Warsaw to Krakow approximately 250km)
        warsaw_lat, warsaw_lon = 52.2297, 21.0122
        krakow_lat, krakow_lon = 50.0647, 19.9450
        
        dist = haversine_distance(warsaw_lat, warsaw_lon, krakow_lat, krakow_lon)
        assert 240 < dist < 260  # Approximate distance
    
    def test_find_closest_timestamp(self):
        """Test finding closest timestamp."""
        available = [
            '2023-05-15 10:00:00',
            '2023-05-15 10:15:00',
            '2023-05-15 10:30:00',
            '2023-05-15 10:45:00'
        ]
        
        # Exact match
        result = find_closest_timestamp('2023-05-15 10:15:00', available)
        assert result == '2023-05-15 10:15:00'
        
        # Closest match
        result = find_closest_timestamp('2023-05-15 10:17:00', available)
        assert result == '2023-05-15 10:15:00'
        
        # Edge case - empty list
        result = find_closest_timestamp('2023-05-15 10:00:00', [])
        assert result is None
    
    @pytest.mark.parametrize("start_date,end_date,should_be_valid", [
        ('2023-05-15', '2023-05-20', True),
        ('2023-05-20', '2023-05-15', False),
        ('invalid', '2023-05-15', False),
    ])
    def test_validate_date_range(self, start_date, end_date, should_be_valid):
        """Test date range validation."""
        valid, error = validate_date_range(start_date, end_date)
        assert valid == should_be_valid
    
    def test_convert_max_bikes_param(self):
        """Test max bikes parameter conversion."""
        assert convert_max_bikes_param('all') is None
        assert convert_max_bikes_param(None) is None
        assert convert_max_bikes_param('100') == 100
        assert convert_max_bikes_param(50) == 50
        assert convert_max_bikes_param('invalid') is None
