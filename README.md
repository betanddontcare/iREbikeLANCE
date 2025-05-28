![iREbikeLANCE Logo](https://i.ibb.co/1tgDLS4X/e36fa2bb-b33b-4491-b034-93130804203f.png)

**Intelligent Rebalancing: Reinforcement Learning Agent for Optimal Bike-Sharing Distribution Powered by Historical Usage Data**
---

**iREbikeLANCE** is a comprehensive platform demonstrating the power of Deep Reinforcement Learning (DRL) for optimizing dynamic bike rebalancing in urban Bike-Sharing Systems (BSS). This project provides an interactive frontend to configure, train, and evaluate a Proximal Policy Optimization (PPO) agent within a custom simulation environment built on real-world historical data from Warsaw's BSS.

The core challenge addressed is the inherent demand imbalance in BSS, where stations frequently become empty or full, leading to poor service reliability and user dissatisfaction. iREbikeLANCE tackles this by training an RL agent to learn intelligent, proactive rebalancing strategies.

---

## 🚀 Key Features

*   **Interactive Frontend (React):**
    *   🗺️ **Dynamic Map Visualization:** Display of bike stations with real-time (snapshot-based) bike counts and capacities using Leaflet.
    *   📅 **Historical Data Exploration:** Select and view BSS snapshots for specific dates and times.
    *   🌦️ **Weather Integration:** View weather conditions corresponding to selected snapshots.
    *   ⚙️ **Customizable Reward Function:** Interactively define and tune weights for various components of the RL agent's reward signal (e.g., penalties for empty/full stations, trip failures, movement costs; bonuses for successful trips, diversity, proactive moves).
    *   🧠 **RL Agent Training Control:** Configure training parameters (number of stations, training steps, learning rate, device, etc.) and initiate the training process.
    *   📊 **Live Training Monitoring:** Track training progress, view real-time logs, and observe key performance metrics (reward, success rate, distribution error).
    *   📈 **Comprehensive Results & Charts:** Analyze detailed training outcomes, including performance metrics and visualizations of learning curves.
    *   ⚖️ **Model Comparison:** Evaluate your trained agent against a baseline model under identical test conditions.
    *   💾 **Model Download:** Download the trained PPO agent for further use or analysis.
*   **Sophisticated Backend (Flask, Python):**
    *   🔩 **PPO-Based RL Agent:** Utilizes Proximal Policy Optimization for stable and efficient learning of rebalancing policies.
    *   🏙️ **Realistic Simulation Environment:** Custom Gym-compliant environment powered by historical BSS data from Warsaw (over 300 stations, extensive trip/weather/attribute data).
    *   📝 **Rich State Representation:** The RL agent observes a detailed state including bike distribution, station capacities, temporal dynamics, weather conditions, station attributes, demand forecasts, and adaptive fill targets.
    *   🌍 **Geospatial Awareness:** Incorporates real-world distances between stations (Haversine formula) to model logistical costs accurately.
    *   🌡️ **Advanced Weather Modeling:** Integrates detailed weather data, impacting simulated user behavior and agent's reward.
    *   🎯 **Adaptive Fill Targets:** Station fill targets are dynamically adjusted based on time of day, day type, historical patterns, and surrounding Points of Interest (POI).
    *   🔧 **Modular Data Processing:** Robust pipelines for loading, cleaning, and transforming diverse datasets.

---

## 💡 Core Innovations & Contributions

This project builds upon and extends concepts from recent research in BSS rebalancing:

1.  **End-to-End RL for Operator-Led Rebalancing:** Directly learns a policy for *where* to move bikes from, *where* to move them to, and *how many* to move, for a large-scale, station-based BSS.
2.  **Highly Granular & Customizable Reward Function:** Enables fine-tuning of the agent's behavior to align with diverse operational goals (service level vs. cost efficiency).
3.  **Integration of Heterogeneous Data:** The agent leverages a rich, multi-dimensional state representation, crucial for learning complex, context-aware strategies.
4.  **Real-World Data & Scalability:** Trained and tested on a comprehensive dataset from a large, operational BSS (Warsaw), demonstrating scalability.
5.  **Proactive & Adaptive Strategies:** The agent learns to anticipate demand and adapt to changing conditions (time, weather, POI context) rather than relying on purely reactive measures.
6.  **Station Overflow Analysis:** Investigates the impact of allowing stations to exceed nominal capacity, revealing practical trade-offs for BSS operators.

---

## 📊 Performance Highlights

Based on extensive simulations using historical data from Warsaw's BSS:

*   **Significant Success Rate Improvement:** The trained PPO agent achieved a trip success rate of **80.60%** on unseen test data, a dramatic improvement from a baseline model's 5.56%.
*   **Efficient Relocations:** Reduced average bike relocation distances by **41%** compared to early-stage models, optimizing logistical costs.
*   **Effective Distribution Management:** Systematically reduced mean distribution error and the occurrence of empty or critically full stations.
*   **Impact of Flexible Capacity:** Allowing a modest 5% station overflow (bikes locked to other bikes) increased the trip success rate by an additional **10.7 percentage points** with minimal impact on operational costs.

---

## 🛠 Technology Stack

*   **Frontend:** React, JavaScript, Tailwind CSS, Leaflet.js, Lucide React (icons)
*   **Backend:** Python, Flask, Stable Baselines3 (PPO), Pandas, NumPy
*   **Simulation:** Custom OpenAI Gym/Gymnasium-compliant environment
*   **Data:** JSON, CSV

---

## 🚀 Getting Started

### Prerequisites

*   Node.js and npm (for frontend)
*   Python 3.9+ and pip (for backend)
*   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/iREbikeLANCE.git
    cd iREbikeLANCE
    ```

2.  **Backend Setup:**
    ```bash
    cd backend
    python -m venv venv
    # Activate virtual environment
    # Windows:
    # venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate
    pip install -r requirements.txt
    # Ensure your data files (trips.json, stations.json, etc.) are in the backend/data directory as per config.py
    ```

3.  **Frontend Setup:**
    ```bash
    cd ../frontend # Assuming you are in the backend directory
    npm install
    ```

### Running the Application

1.  **Start the Backend Server:**
    Navigate to the `backend` directory and run:
    ```bash
    # Ensure your virtual environment is activated
    python app.py
    ```
    The backend API will typically be available at `http://localhost:5000`.

2.  **Start the Frontend Development Server:**
    Navigate to the `frontend` directory and run:
    ```bash
    npm start
    ```
    The frontend will typically open automatically in your browser at `http://localhost:3000`.

---

## 📖 Usage Guide

1.  **Explore Data:** Use the map and date/time pickers on the main screen to visualize historical station states and weather conditions.
2.  **Configure Reward Weights:** Navigate to the "Reward Function Configuration" section in the sidebar. Adjust the weights for each component to guide the agent's learning. Reset to research-backed optimal weights if needed.
3.  **Set Training Parameters:** In the "Training Configuration" section, define parameters like the number of stations, training steps, bikes/users limit, device, and training/testing date ranges.
4.  **Start Training:** Click the "Start Training with Custom Rewards" button.
5.  **Monitor Progress:** Observe the training progress bar, live console logs, and real-time metrics (timestep, episodes, average reward, loss, elapsed time). Training charts will populate as data becomes available.
6.  **Analyze Results:** Once training is complete, the "Training Results" section will display key performance indicators like final success rate, average reward, total relocations, and training time. Detailed training charts provide insights into the learning process.
7.  **Compare Models:** Click "Compare vs Baseline" to evaluate your trained model against a non-learning baseline agent using the same test data and parameters.
8.  **Download Model:** If satisfied with the training, download the `*.zip` file containing the trained RL model.

---

## 👨‍🔬 Authors & Acknowledgements

This project is based on research conducted by:

*   **Dr. Eng. Igor Betkier** (Project Coordinator & System Architecture)

Affiliated with the Faculty of Civil Engineering and Geodesy, Military University of Technology, Warsaw, Poland.

The UI and simulation environment are inspired by the findings and methodologies presented in their research paper "Intelligent Rebalancing: Reinforcement Learning Agent for Optimal Bike-Sharing Distribution Powered by Historical Usage Data".
[Read the paper on ResearchGate](https://www.researchgate.net/publication/391807731_Intelligent_Rebalancing_Reinforcement_Learning_Agent_for_Optimal_Bike-Sharing_Distribution_Powered_by_Historical_Usage_Data)


