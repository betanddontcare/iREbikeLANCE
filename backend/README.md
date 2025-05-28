# iREbikeLANCE – Backend

The **iREbikeLANCE** backend module supports intelligent management and rebalancing of bike-sharing systems. It includes core logic for data preprocessing, environment simulation for reinforcement learning, API endpoints, and utilities for real-time decision making.

## 🛠 Technologies

- Python 3.11
- Flask – REST API framework
- PyTorch – Deep learning
- NumPy, Pandas – Data handling
- pytest – Testing framework
- OpenAI Gym – RL environment interface (custom)

## 🚀 Getting Started

### Prerequisites

Make sure Python 3.11+ is installed. It is recommended to use a virtual environment:

python -m venv .venv
.\.venv\Scripts\activate  # On Windows

### Install dependencies

pip install -r requirements.txt

### Run the backend server

python app.py
The API will be available at:
http://127.0.0.1:5000

## 📡 API Endpoints (Examples)

Method	Endpoint	Description
GET	    /health	Health check
POST	/predict_demand	Predicts demand based on features
POST	/simulate_action	Simulates an RL relocation action

Example: POST /predict_demand

### Request:

{
  "station_id": "ST123",
  "features": {
    "hour": 8,
    "day_of_week": 2,
    "temp": 17.5,
    "weather": "clear"
  }
}

### Response:

{
  "predicted_trips": 3.42
}

## 🧪 Testing

All tests are located in the tests/ folder.

### PowerShell

$env:PYTHONPATH = "."
pytest

## 📁 Project Structure

backend/
├── app.py                  # Entry point (Flask server)
├── config.py               # Global configuration
├── data_processing.py      # Feature engineering & preprocessing
├── environment.py          # Reinforcement learning environment
├── utils.py                # Utility functions
├── training.py             # Training functions
├── tests/                  # Unit tests
│   ├── test_app.py
│   └── test_*.py
├── requirements.txt
└──  README.md

👨‍🔬 Authors

Developed by: Dr. Eng. Igor Betkier – Project coordinator & system architecture

📬 Contact

For collaboration or technical inquiries, please contact:
📧 igor.betkier@wat.edu.pl
🌍 https://github.com/betanddontcare