"""Test Flask API endpoints."""

import pytest
import json
from unittest.mock import patch

@pytest.fixture
def client():
    """Create test client for Flask app."""
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestFlaskAPI:
    """Test Flask API endpoints."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_system_status_endpoint(self, client):
        """Test system status endpoint."""
        response = client.get('/api/system/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'backend_ready' in data
        assert 'training_active' in data
    
    def test_optimal_reward_weights_endpoint(self, client):
        """Test optimal reward weights endpoint."""
        response = client.get('/api/training/reward_weights')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'optimal_weights' in data
        assert 'description' in data