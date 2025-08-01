# iREbikeLANCE API Documentation

## Backend Endpoints

### System Status
- `GET /api/system/status` - Get system health and status
- `GET /api/data_info` - Get data loading status

### Data Endpoints  
- `GET /api/available_timestamps` - List available data snapshots
- `GET /api/stations?timestamp={timestamp}` - Get station states
- `GET /api/weather` - Get weather data

### Training Endpoints
- `POST /api/train` - Start training with custom configuration
- `GET /api/training/status` - Get current training progress
- `POST /api/training/cancel` - Cancel ongoing training
- `GET /api/training/reward_weights` - Get optimal reward weights

### Model Management
- `POST /api/training/compare_models` - Compare trained vs baseline
- `GET /api/model/download?path={path}` - Download trained model
