"""
Bike sharing environment implementation for reinforcement learning.
"""

import gymnasium as gym
import numpy as np
import logging
from gymnasium import spaces
from numba import njit
from config import OPTIMAL_REWARD_WEIGHTS

logger = logging.getLogger(__name__)

@njit
def simulate_trips(bike_counts, trips, station_caps):
    """Simulate bike trips using numba for speed."""
    n_trips_completed = 0
    n_trips_failed = 0

    for i in range(trips.shape[0]):
        origin = trips[i, 0]
        dest = trips[i, 1]
        if bike_counts[origin] > 0 and bike_counts[dest] < station_caps[dest]:
            bike_counts[origin] -= 1
            bike_counts[dest] += 1
            n_trips_completed += 1
        else:
            n_trips_failed += 1

    return bike_counts, n_trips_completed, n_trips_failed

class BikeSharingEnv(gym.Env):
    """Bike sharing rebalancing environment using Gymnasium interface."""
    
    def __init__(
        self,
        grouped_trips,
        grouped_bike_counts,
        station_uids,
        station_capacity=None,
        station_attributes=None,
        station_attributes_path=None,
        weather_data=None,
        max_bikes=50,
        time_step_minutes=15,
        max_episode_length=96,
        debug=False,
        adaptive_targets=True,
        station_hourly_patterns=None,
        reward_weights=None,
        distance_matrix=None,
        station_info_json=None
    ):
        super().__init__()
        self.debug = debug
        self.grouped_trips = grouped_trips
        self.grouped_bike_counts = grouped_bike_counts
        self.station_uids = station_uids
        self.num_stations = len(station_uids)
        self.adaptive_targets = adaptive_targets
        self.station_hourly_patterns = station_hourly_patterns
        self.last_from_station = None

        # Handle distance matrix
        self.distance_matrix = distance_matrix
        if self.distance_matrix is None:
            self.distance_matrix = np.ones((self.num_stations, self.num_stations), dtype=np.float32)
            np.fill_diagonal(self.distance_matrix, 0)

        # Reward weights - use provided or optimal defaults
        if reward_weights is None:
            self.reward_weights = OPTIMAL_REWARD_WEIGHTS.copy()
        else:
            self.reward_weights = reward_weights

        # Set station capacities
        self.station_capacity = {}
        if station_capacity:
            for s_id in station_uids:
                if s_id in station_capacity:
                    self.station_capacity[s_id] = station_capacity[s_id]
                else:
                    self.station_capacity[s_id] = float(max_bikes)
        else:
            for s_id in station_uids:
                self.station_capacity[s_id] = float(max_bikes)

        # Set station attributes
        self.station_attributes = {}
        if station_attributes:
            for s_id in station_uids:
                if s_id in station_attributes:
                    self.station_attributes[s_id] = station_attributes[s_id]
                else:
                    self.station_attributes[s_id] = {'d_city_cen': 0.0, 'pop_2023': 0.0}
        else:
            for s_id in station_uids:
                self.station_attributes[s_id] = {'d_city_cen': 0.0, 'pop_2023': 0.0}

        self.weather_data = weather_data if weather_data else {}
        self.max_bikes = max_bikes
        self.time_step_minutes = time_step_minutes
        self.max_episode_length = max_episode_length

        # Initialize state
        self.time_steps = sorted(list(self.grouped_trips.keys()))
        self.total_steps = len(self.time_steps)
        self.current_step = 0
        self.episode_start_step = 0
        self.completed_trips = 0
        self.failed_trips = 0
        self.moved_bikes_history = {}
        self.action_count = {}

        # Map station UIDs to indices
        self.uid_to_index = {uid: i for i, uid in enumerate(self.station_uids)}
        self.index_to_uid = {i: uid for i, uid in enumerate(self.station_uids)}

        # Default target state
        self.target_fraction = 0.5
        self.station_target = np.array([self.station_capacity[uid] * self.target_fraction
                                       for uid in self.station_uids], dtype=np.float32)

        # Precompute vectorized trips for numba acceleration
        self.grouped_trips_vectorized = {}
        for ts in self.time_steps:
            trips = self.grouped_trips.get(ts, [])
            origins = []
            destinations = []
            for trip in trips:
                if 'from_station' not in trip or 'to_station' not in trip:
                    continue

                origin = self.uid_to_index.get(trip.get('from_station'))
                dest = self.uid_to_index.get(trip.get('to_station'))
                if origin is not None and dest is not None:
                    origins.append(origin)
                    destinations.append(dest)

            if origins and destinations:
                try:
                    self.grouped_trips_vectorized[ts] = np.array(list(zip(origins, destinations)), dtype=np.int64)
                except Exception as e:
                    logger.error(f"Error processing trips for timestamp {ts}: {e}")
                    self.grouped_trips_vectorized[ts] = np.empty((0, 2), dtype=np.int64)
            else:
                self.grouped_trips_vectorized[ts] = np.empty((0, 2), dtype=np.int64)

        # Precompute capacity array for numba acceleration
        self.station_caps = np.array([self.station_capacity.get(uid, self.max_bikes)
                                      for uid in self.station_uids], dtype=np.float32)

        # Observation space
        obs_dim = (
            self.num_stations * 2 +    # Raw + normalized bike counts
            1 +                        # Current step
            4 +                        # Weather features
            3 +                        # Time features
            10 * self.num_stations +   # Station attributes
            self.num_stations * 2      # Demand + targets
        )

        self.observation_space = spaces.Box(
            low=-9999, high=9999,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([self.num_stations, self.num_stations, 20])

        self.bike_counts = None
        self.reset()

    def _update_targets_by_time(self, hour, is_weekend):
        """Update target bike counts based on time of day and day type."""
        if is_weekend:
            if hour < 8:
                base_fraction = 0.6
            elif 8 <= hour < 12:
                base_fraction = 0.5
            elif 12 <= hour < 18:
                base_fraction = 0.4
            else:
                base_fraction = 0.5
        else:
            if hour < 6:
                base_fraction = 0.6
            elif 6 <= hour < 10:
                base_fraction = 0.3
            elif 16 <= hour < 20:
                base_fraction = 0.7
            else:
                base_fraction = 0.5

        for i in range(self.num_stations):
            self.station_target[i] = self.station_caps[i] * base_fraction

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.episode_start_step = 0
        self.completed_trips = 0
        self.failed_trips = 0
        self.moved_bikes_history = {}
        self.action_count = {}
        self.last_from_station = None

        # Initialize bike counts from data
        first_ts = self.time_steps[self.current_step]
        init_dict = self.grouped_bike_counts.get(first_ts, {})

        self.bike_counts = np.zeros(self.num_stations, dtype=np.float32)
        for uid, val in init_dict.items():
            idx = self.uid_to_index.get(uid, None)
            if idx is not None:
                max_cap = self.station_capacity.get(uid, self.max_bikes)
                self.bike_counts[idx] = min(val, max_cap)

        # Update targets
        hour = first_ts.hour if first_ts else 0
        is_weekend = first_ts.weekday() >= 5 if first_ts else False

        if self.adaptive_targets:
            self._update_targets_by_time(hour, is_weekend)

        obs = self._get_obs()
        if self.debug:
            logger.info(f"[RESET] => Bike counts initialized for {len(self.bike_counts)} stations")

        return obs, {}

    def step(self, action):
        """Execute one environment step."""
        from_station, to_station, num_bikes = action.astype(int)
        self.last_from_station = from_station

        if from_station == to_station:
            num_bikes = 0

        bikes_from_before = float(self.bike_counts[from_station])
        bikes_to_before = float(self.bike_counts[to_station])

        uid_to = self.station_uids[to_station]
        cap_to = self.station_capacity.get(uid_to, self.max_bikes)

        bikes_from = self.bike_counts[from_station]
        can_fit = cap_to - self.bike_counts[to_station]
        bikes_moved = max(0, min(num_bikes, bikes_from, can_fit))

        # Move bikes
        self.bike_counts[from_station] -= bikes_moved
        self.bike_counts[to_station] += bikes_moved

        bikes_from_after = float(self.bike_counts[from_station])
        bikes_to_after = float(self.bike_counts[to_station])

        # Record station pair for diversity bonus
        station_pair = (from_station, to_station)
        if bikes_moved > 0:
            if station_pair in self.moved_bikes_history:
                self.moved_bikes_history[station_pair] += bikes_moved
            else:
                self.moved_bikes_history[station_pair] = bikes_moved

        # Advance time step
        self.current_step += 1
        done = False

        # Simulate trips for this time step
        trips_completed = 0
        trips_failed = 0

        if self.current_step < self.total_steps:
            ts = self.time_steps[self.current_step]
            trips_vec = self.grouped_trips_vectorized.get(ts, np.empty((0, 2), dtype=np.int64))
            if trips_vec.shape[0] > 0:
                self.bike_counts, trips_completed, trips_failed = simulate_trips(
                    self.bike_counts, trips_vec, self.station_caps)

            hour = ts.hour if ts else 0
            is_weekend = ts.weekday() >= 5 if ts else False

            if self.adaptive_targets:
                self._update_targets_by_time(hour, is_weekend)
        else:
            done = True

        if (self.current_step - self.episode_start_step) >= self.max_episode_length:
            done = True

        self.completed_trips += trips_completed
        self.failed_trips += trips_failed

        # Calculate reward using configurable weights
        reward = self.reward_weights['step_penalty']

        # Distribution error
        normalized_bike_counts = self.bike_counts / self.station_caps
        normalized_targets = self.station_target / self.station_caps
        dist_error = np.mean((normalized_bike_counts - normalized_targets) ** 2)
        reward += self.reward_weights['dist_error_coef'] * dist_error

        # Count empty and full stations
        empty_stations = np.sum(self.bike_counts == 0)
        full_stations = np.sum(self.bike_counts >= self.station_caps * 0.95)

        empty_ratio = empty_stations / self.num_stations
        full_ratio = full_stations / self.num_stations

        reward += self.reward_weights['empty_station_penalty'] * empty_ratio
        reward += self.reward_weights['full_station_penalty'] * full_ratio

        # Cost of moving bikes
        if bikes_moved > 0:
            distance = self.distance_matrix[from_station, to_station]
            distance_cost = abs(self.reward_weights['distance_cost_factor']) * distance * bikes_moved
            reward += self.reward_weights['distance_cost_factor'] * distance_cost

        # Reward successful trips
        if trips_completed > 0:
            reward += self.reward_weights['successful_trip_bonus'] * trips_completed

        # Penalize failed trips
        if trips_failed > 0:
            failure_rate = trips_failed / max(1, trips_completed + trips_failed)
            reward += self.reward_weights['trip_failure_penalty'] * failure_rate

        obs = self._get_obs()

        info = {
            "from_station_id": self.station_uids[from_station],
            "to_station_id": self.station_uids[to_station],
            "bikes_moved": int(bikes_moved),
            "bikes_from_before": bikes_from_before,
            "bikes_from_after": bikes_from_after,
            "bikes_to_before": bikes_to_before,
            "bikes_to_after": bikes_to_after,
            "trips_completed": trips_completed,
            "trips_failed": trips_failed,
            "dist_error": float(dist_error),
            "empty_stations": int(empty_stations),
            "full_stations": int(full_stations),
            "reward": float(reward),
            "move_distance": float(self.distance_matrix[from_station, to_station]) if bikes_moved > 0 else 0.0
        }

        if self.debug and (self.current_step % 10 == 0 or bikes_moved > 0):
            logger.info(f"[STEP {self.current_step}] from={from_station}, to={to_station}, moved={bikes_moved}, reward={reward:.3f}")

        return obs, reward, done, False, info

    def _get_obs(self):
        """Construct observation vector for the agent."""
        obs_list = []

        # 1. Bike counts at each station
        obs_list.extend(list(self.bike_counts))

        # 2. Normalized bike counts
        normalized_counts = self.bike_counts / self.station_caps
        obs_list.extend(normalized_counts)

        # 3. Current step normalized
        obs_list.append(float(self.current_step) / self.max_episode_length)

        # 4. Weather and time features (simplified)
        if self.current_step < len(self.time_steps):
            ts = self.time_steps[self.current_step]
            hour = ts.hour + ts.minute / 60.0
            sin_hour = np.sin(2 * np.pi * hour / 24)
            cos_hour = np.cos(2 * np.pi * hour / 24)
            is_weekend = 1.0 if ts.weekday() >= 5 else 0.0
        else:
            sin_hour, cos_hour = (0.0, 0.0)
            is_weekend = 0.0

        obs_list.extend([15.0, 2, 0.0, 10.0])  # Simple weather placeholder
        obs_list.extend([sin_hour, cos_hour, is_weekend])

        # 5. Station attributes (simplified)
        for uid in self.station_uids:
            st_dict = self.station_attributes.get(uid, {})
            obs_list.extend([
                st_dict.get('d_city_cen', 0.0),
                st_dict.get('pop_2023', 0.0),
                st_dict.get('d_metro_st', 0.0),
                st_dict.get('d_bus_tram', 0.0),
                st_dict.get('d_railway', 0.0),
                st_dict.get('d_school', 0.0),
                st_dict.get('d_academy', 0.0),
                st_dict.get('d_shop_cen', 0.0),
                st_dict.get('d_health', 0.0),
                st_dict.get('d_sport', 0.0)
            ])

        # 6. Expected demand (simplified)
        expected_demand = np.zeros(self.num_stations, dtype=np.float32)
        obs_list.extend(expected_demand)

        # 7. Target bike counts
        obs_list.extend(self.station_target)

        return np.array(obs_list, dtype=np.float32)