"""
Data loading and processing utilities for bike sharing RL application.
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import WEATHER_MAPPING
from utils import get_data_file_path

logger = logging.getLogger(__name__)

def load_station_ids_from_bikes_file(json_bike_counts_path):
    """Extract station IDs from the bike counts file."""
    if not os.path.exists(json_bike_counts_path):
        raise FileNotFoundError(f"File not found: {json_bike_counts_path}")
    
    with open(json_bike_counts_path, 'r', encoding='utf-8') as f:
        bike_counts_data = json.load(f)
    
    first_timestamp = next(iter(bike_counts_data))
    stations_info = bike_counts_data[first_timestamp]
    station_ids = set(stations_info.keys())
    return station_ids

def build_station_capacity_mapping(json_bike_counts_path, json_stations_info_path):
    """Create mapping between station IDs and their capacities."""
    station_ids = load_station_ids_from_bikes_file(json_bike_counts_path)
    
    with open(json_stations_info_path, 'r', encoding='utf-8') as f:
        stations_data_obj = json.load(f)
    
    if 'stations_data' not in stations_data_obj:
        raise ValueError("File does not contain 'stations_data' key")
    
    capacity_dict = {}
    for station_info in stations_data_obj['stations_data']:
        uid = station_info.get('uid', None)
        if uid is None:
            continue
        availability = station_info.get('availabilityStatus', {})
        bike_racks = availability.get('bikeRacks', 0)
        capacity_dict[str(uid)] = float(bike_racks)
    
    station_capacity = {}
    for st_id in station_ids:
        if st_id in capacity_dict:
            station_capacity[st_id] = capacity_dict[st_id]
        else:
            station_capacity[st_id] = 20.0  # Default capacity
    
    return station_capacity

def load_station_attributes_extended(geojson_path, valid_station_ids):
    """Load extended attributes for stations from GeoJSON file."""
    if not os.path.exists(geojson_path):
        return {}
        
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
        
    features = geojson_data.get('features', [])
    station_attributes = {}
    
    for feat in features:
        props = feat.get('properties', {})
        uid = props.get('uid', None)
        if uid is None:
            continue
            
        uid_str = str(uid)
        if uid_str not in valid_station_ids:
            continue
            
        attributes = {}
        for key, value in props.items():
            if key not in ['uid', 'lat', 'lon']:
                try:
                    attributes[key] = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    attributes[key] = 0.0
        
        station_attributes[uid_str] = attributes
        
    return station_attributes

def normalize_station_attributes(station_attributes):
    """Normalize station attributes to range [0, 1]."""
    attribute_values = {}
    
    for station_id, attrs in station_attributes.items():
        for attr_name, attr_value in attrs.items():
            if attr_name not in attribute_values:
                attribute_values[attr_name] = []
            attribute_values[attr_name].append(attr_value)
    
    attr_mins = {}
    attr_maxs = {}
    
    for attr_name, values in attribute_values.items():
        attr_mins[attr_name] = min(values)
        attr_maxs[attr_name] = max(values)
    
    normalized_attributes = {}
    
    for station_id, attrs in station_attributes.items():
        normalized_attributes[station_id] = {}
        
        for attr_name, attr_value in attrs.items():
            attr_min = attr_mins[attr_name]
            attr_max = attr_maxs[attr_name]
            
            if attr_max > attr_min:
                normalized_value = (attr_value - attr_min) / (attr_max - attr_min)
            else:
                normalized_value = 0.5
                
            normalized_attributes[station_id][attr_name] = normalized_value
    
    return normalized_attributes

def load_and_process_data_with_date_range(
    json_trips_path,
    json_bike_counts_path,
    station_capacity_map=None,
    station_attributes=None,
    station_attributes_path=None,
    weather_data=None,
    time_step_minutes=15,
    top_n_stations=100,
    debug_mode=False,
    start_date=None,
    end_date=None,
    max_bikes_users=None,
    status_queue=None
):
    """Load and process data for bike sharing environment with specific date range."""
    if not os.path.exists(json_trips_path):
        raise FileNotFoundError(f"Trip file does not exist: {json_trips_path}")
    if not os.path.exists(json_bike_counts_path):
        raise FileNotFoundError(f"Bike counts file does not exist: {json_bike_counts_path}")

    logger.info(f"Loading trips data from {json_trips_path}...")
    if status_queue:
        status_queue.put({
            'status': 'loading_trips',
            'progress': 15,
            'message': 'Loading bike trips dataset...'
        })
    
    with open(json_trips_path, 'r', encoding='utf-8') as f:
        trips_data = json.load(f)

    # Limit bikes/users if specified
    total_bikes_available = len(trips_data)
    logger.info(f"Total bikes/users available in trips data: {total_bikes_available}")

    if max_bikes_users and max_bikes_users < total_bikes_available:
        logger.info(f"Limiting to {max_bikes_users} most active bikes/users (from {total_bikes_available})")
        if status_queue:
            status_queue.put({
                'status': 'filtering_bikes',
                'progress': 22,
                'message': f'Analyzing activity of {total_bikes_available} bikes...'
            })
        
        bike_trip_counts = {bike_id: len(trips) for bike_id, trips in trips_data.items()}
        sorted_bikes = sorted(bike_trip_counts.items(), key=lambda x: x[1], reverse=True)
        
        selected_bike_ids = [bike_id for bike_id, _ in sorted_bikes[:max_bikes_users]]
        filtered_trips_data = {bike_id: trips_data[bike_id] for bike_id in selected_bike_ids}
        
        total_trips_before = sum(len(trips) for trips in trips_data.values())
        total_trips_after = sum(len(trips) for trips in filtered_trips_data.values())
        
        logger.info(f"Filtered trips data:")
        logger.info(f"  Selected {len(selected_bike_ids)} bikes from {total_bikes_available}")
        logger.info(f"  Total trips: {total_trips_after} (from {total_trips_before})")
        logger.info(f"  Data reduction: {(1 - total_trips_after/total_trips_before)*100:.1f}%")
        
        if status_queue:
            reduction_pct = (1 - total_trips_after/total_trips_before)*100
            status_queue.put({
                'status': 'bikes_selected',
                'progress': 30,
                'message': f'Dataset filtered: {total_trips_after} trips from {len(selected_bike_ids)} bikes ({reduction_pct:.1f}% reduction)'
            })
        
        trips_data = filtered_trips_data
    else:
        total_trips = sum(len(trips) for trips in trips_data.values())
        logger.info(f"Using all {total_bikes_available} bikes with {total_trips} total trips")

    # Process trips
    trips_list = []
    for bike_id, bike_trips in trips_data.items():
        for trip in bike_trips:
            dtime = pd.to_datetime(trip['departure_time'])
            trips_list.append({
                'timestamp': dtime,
                'from_station': trip['from'],
                'to_station': trip['to']
            })

    df_trips = pd.DataFrame(trips_list)
    logger.info(f"Processed {len(df_trips)} individual trips from selected bikes")
    df_trips.sort_values('timestamp', inplace=True)
    
    if status_queue:
        status_queue.put({
            'status': 'processing_trips',
            'progress': 40,
            'message': f'Processed {len(df_trips)} trips, identifying top stations...'
        })
    
    df_trips.reset_index(drop=True, inplace=True)

    # Identify most popular stations
    station_counts = df_trips['from_station'].value_counts() + df_trips['to_station'].value_counts()
    top_stations = station_counts.head(top_n_stations).index.tolist()
    
    if status_queue:
        status_queue.put({
            'status': 'filtering_stations', 
            'progress': 45,
            'message': f'Selected top {len(top_stations)} stations, filtering trips...'
        })
    
    logger.info(f"After filtering to top {top_n_stations} stations: {len(df_trips)} trips remaining")

    # Filter trips to only include top stations
    df_trips = df_trips[
        df_trips['from_station'].isin(top_stations) &
        df_trips['to_station'].isin(top_stations)
    ]

    logger.info(f"Loading bike counts data from {json_bike_counts_path}...")
    if status_queue:
        status_queue.put({
            'status': 'loading_bike_counts',
            'progress': 50, 
            'message': 'Loading historical bike availability data...'
        })
    
    with open(json_bike_counts_path, 'r', encoding='utf-8') as f:
        bike_counts_data = json.load(f)

    # Process bike counts
    bike_records = []
    for tstr, stations_dict in bike_counts_data.items():
        tstamp = pd.to_datetime(tstr)
        for uid, val in stations_dict.items():
            bike_records.append({
                'timestamp': tstamp,
                'station_uid': uid,
                'available_bikes': val
            })

    df_bike = pd.DataFrame(bike_records)
    df_bike.sort_values('timestamp', inplace=True)
    
    if status_queue:
        status_queue.put({
            'status': 'processing_bike_counts',
            'progress': 55,
            'message': f'Processed {len(df_bike)} bike count records, creating time bins...'
        })
    
    df_bike.reset_index(drop=True, inplace=True)

    # Determine date range for simulation
    if start_date and end_date:
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        logger.info(f"Using specified date range: {start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')}")

        # Filter dataframes to the specified date range
        df_trips = df_trips[(df_trips['timestamp'] >= start_ts) & (df_trips['timestamp'] <= end_ts)]
        df_bike = df_bike[(df_bike['timestamp'] >= start_ts) & (df_bike['timestamp'] <= end_ts)]
    else:
        start_ts = min(df_trips['timestamp'].min(), df_bike['timestamp'].min())
        end_ts = max(df_trips['timestamp'].max(), df_bike['timestamp'].max())
        logger.info(f"Using full date range: {start_ts} to {end_ts}")

    # Create time bins
    min_floor = start_ts.floor(f'{time_step_minutes}min')
    max_ceil = end_ts.ceil(f'{time_step_minutes}min')
    time_range = pd.date_range(min_floor, max_ceil, freq=f'{time_step_minutes}min')
    
    if status_queue:
        status_queue.put({
            'status': 'creating_time_bins',
            'progress': 60,
            'message': f'Created {len(time_range)} time bins, grouping trip data...'
        })
    
    # Group trips by time step
    grouped_trips = {ts: [] for ts in time_range}
    df_trips['time_step'] = df_trips['timestamp'].dt.floor(f'{time_step_minutes}min')
    for _, row in df_trips.iterrows():
        if row['time_step'] in grouped_trips:
            grouped_trips[row['time_step']].append(row.to_dict())
    
    if status_queue:
        status_queue.put({
            'status': 'grouping_data',
            'progress': 70,
            'message': 'Grouped trips by time intervals, processing bike counts...'
        })
    
    # Group bike counts by time step
    grouped_bikes = {ts: {} for ts in time_range}
    df_bike['time_step'] = df_bike['timestamp'].dt.floor(f'{time_step_minutes}min')
    for _, row in df_bike.iterrows():
        if row['time_step'] in grouped_bikes:
            grouped_bikes[row['time_step']][row['station_uid']] = row['available_bikes']

    station_uids = sorted(list(set(top_stations)))
    logger.info(f"Number of unique stations: {len(station_uids)}")

    # Calculate trip patterns for each station by hour
    station_hourly_patterns = {}
    logger.info("Analyzing hourly trip patterns...")
    
    if status_queue:
        status_queue.put({
            'status': 'analyzing_patterns',
            'progress': 80,
            'message': f'Analyzing hourly trip patterns for {len(station_uids)} stations...'
        })
    
    df_trips['weekday'] = df_trips['timestamp'].dt.weekday < 5

    for station_id in station_uids:
        weekday_data = {}
        weekend_data = {}

        for hour in range(24):
            # Weekday patterns
            weekday_trips_from = df_trips[(df_trips['from_station'] == station_id) &
                                       (df_trips['timestamp'].dt.hour == hour) &
                                       (df_trips['weekday'] == True)]
            weekday_trips_to = df_trips[(df_trips['to_station'] == station_id) &
                                     (df_trips['timestamp'].dt.hour == hour) &
                                     (df_trips['weekday'] == True)]

            weekday_outflow = len(weekday_trips_from)
            weekday_inflow = len(weekday_trips_to)
            weekday_ratio = 0.5

            if weekday_outflow + weekday_inflow > 0:
                weekday_ratio = weekday_outflow / (weekday_outflow + weekday_inflow)

            weekday_data[hour] = {
                'outflow': weekday_outflow,
                'inflow': weekday_inflow,
                'ratio': weekday_ratio
            }

            # Weekend patterns
            weekend_trips_from = df_trips[(df_trips['from_station'] == station_id) &
                                       (df_trips['timestamp'].dt.hour == hour) &
                                       (df_trips['weekday'] == False)]
            weekend_trips_to = df_trips[(df_trips['to_station'] == station_id) &
                                     (df_trips['timestamp'].dt.hour == hour) &
                                     (df_trips['weekday'] == False)]

            weekend_outflow = len(weekend_trips_from)
            weekend_inflow = len(weekend_trips_to)
            weekend_ratio = 0.5

            if weekend_outflow + weekend_inflow > 0:
                weekend_ratio = weekend_outflow / (weekend_outflow + weekend_inflow)

            weekend_data[hour] = {
                'outflow': weekend_outflow,
                'inflow': weekend_inflow,
                'ratio': weekend_ratio
            }

        station_hourly_patterns[station_id] = {
            'weekday': weekday_data,
            'weekend': weekend_data
        }

    # Load station attributes if path provided
    if station_attributes_path and os.path.exists(station_attributes_path):
        logger.info(f"Loading extended station attributes from {station_attributes_path}...")
        loaded_attrs = load_station_attributes_extended(station_attributes_path, set(station_uids))
        station_attributes = normalize_station_attributes(loaded_attrs)
        logger.info(f"Loaded attributes for {len(station_attributes)} stations.")

    # Print data statistics
    total_trips = sum(len(trips) for trips in grouped_trips.values())
    logger.info(f"Total trips in selected date range: {total_trips}")

    days_in_range = (end_ts - start_ts).days + 1
    avg_trips_per_day = total_trips / days_in_range
    logger.info(f"Average trips per day: {avg_trips_per_day:.1f}")

    return (grouped_trips, grouped_bikes, station_uids,
            station_capacity_map, station_attributes, weather_data,
            station_hourly_patterns)

def load_weather_data(weather_csv_path):
    """Load and process weather data from CSV file."""
    if not os.path.exists(weather_csv_path):
        logger.warning(f"Weather data file not found: {weather_csv_path}")
        return pd.DataFrame()
    
    try:
        weather_df = pd.read_csv(weather_csv_path)
        
        if 'timestamp' in weather_df.columns:
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        
        if 'condition' in weather_df.columns:
            weather_df['category'] = weather_df['condition'].astype(str).map(
                lambda x: WEATHER_MAPPING.get(x, 5)
            )
        
        logger.info(f"Loaded {len(weather_df)} weather records")
        return weather_df
        
    except Exception as e:
        logger.error(f"Error loading weather data: {e}")
        return pd.DataFrame()