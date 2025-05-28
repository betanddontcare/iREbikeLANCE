import React, { useState, useEffect, useRef } from 'react';
import { Play, Download, BarChart3, Settings, MapPin, Brain, Zap, Database, Cloud, Clock, AlertCircle, CheckCircle, Loader, X, CalendarDays, PauseCircle, TrendingUp, Sliders, RefreshCw } from 'lucide-react';

const BikeRebalancingApp = () => {
  // State management
  const [selectedStation, setSelectedStation] = useState(null);
  const [stations, setStations] = useState([]);
  const [weatherData, setWeatherData] = useState([]);
  const [selectedDateTime, setSelectedDateTime] = useState('');
  const [selectedDate, setSelectedDate] = useState('2023-11-16');
  const [selectedTime, setSelectedTime] = useState('09:00');
  const [availableSnapshots, setAvailableSnapshots] = useState([]);
  const [snapshotResponseInfo, setSnapshotResponseInfo] = useState(null);
  const logsContainerRef = useRef(null);
  const [timestampInfo, setTimestampInfo] = useState(null);
  
  const [trainingDateRange, setTrainingDateRange] = useState({ 
    start: '2023-05-15', 
    end: '2023-11-15' 
  });
  
  const [testingDateRange, setTestingDateRange] = useState({
    start: '2023-11-16',
    end: '2023-11-22'
  });
  
  const [dataInfo, setDataInfo] = useState({});
  const [backendLoading, setBackendLoading] = useState(true);

  const [modelComparison, setModelComparison] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const [showComparison, setShowComparison] = useState(false);

  const [rewardWeights, setRewardWeights] = useState({
    stepPenalty: -0.003,
    distErrorCoef: -0.12,
    moveCostCoef: -0.003,
    emptyStationPenalty: -0.25,
    fullStationPenalty: -0.25,
    tripFailurePenalty: -1.5,
    diversityBonus: 0.03,
    successfulTripBonus: 0.02,
    proactiveBonus: 0.01,
    distanceCostFactor: -0.002
  });

  const [trainingConfig, setTrainingConfig] = useState({
    numStations: 100,
    steps: 100000,
    device: 'auto',
    seed: 42,
    algorithm: 'PPO',
    learningRate: 0.0003,
    batchSize: 64,
    selectedSnapshotTimestamp: '',
    maxBikes: 'all',
  });

  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingResults, setTrainingResults] = useState(null);
  const [showResults, setShowResults] = useState(false);

  const [trainingLogs, setTrainingLogs] = useState([]);
  const [trainingMetrics, setTrainingMetrics] = useState({
    timestep: 0, episodes: 0, loss: null, learning_rate: null, ep_rew_mean: null, elapsed_time: 0
  });
  const [trainingCharts, setTrainingCharts] = useState(null);
  const [modelPath, setModelPath] = useState(null);
  const [canCancelTraining, setCanCancelTraining] = useState(false);

  // Store previous stations to prevent map refresh during training
  const [cachedStations, setCachedStations] = useState([]);

  // FIXED: Navigate snapshot properly updates map
  const navigateSnapshot = (direction) => {
    if (!availableSnapshots.length) return;
    
    const currentIndex = availableSnapshots.findIndex(
      snapshot => snapshot.timestamp === selectedDateTime
    );
    
    let newIndex;
    if (direction === 'next') {
      newIndex = currentIndex < availableSnapshots.length - 1 ? currentIndex + 1 : 0;
    } else {
      newIndex = currentIndex > 0 ? currentIndex - 1 : availableSnapshots.length - 1;
    }
    
    const newTimestamp = availableSnapshots[newIndex].timestamp;
    const [datePart, timePart] = newTimestamp.split(' ');
    const [hours, minutes] = timePart.split(':');
    
    // Update all state components synchronously
    setSelectedDate(datePart);
    setSelectedTime(`${hours}:${minutes}`);
    setSelectedDateTime(newTimestamp);
    setTrainingConfig(prevConfig => ({...prevConfig, selectedSnapshotTimestamp: newTimestamp}));
    
    console.log(`Navigation: ${direction} -> ${newTimestamp}`);
  };

  // LeafletWarsawMap component with fixes
  const LeafletWarsawMap = ({ stations = [], selectedStation, onStationClick }) => {
    const mapRef = useRef(null);
    const [leafletMap, setLeafletMap] = useState(null);
    const [markersLayer, setMarkersLayer] = useState(null);
    const mapInstanceRef = useRef(null);

    const safeStations = Array.isArray(stations) ? stations : [];

    useEffect(() => {
      const initMap = async () => {
          if (!mapRef.current || mapInstanceRef.current) return;
          
          if (!document.getElementById('leaflet-css')) {
              const link = document.createElement('link');
              link.id = 'leaflet-css';
              link.rel = 'stylesheet';
              link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
              link.integrity = 'sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=';
              link.crossOrigin = '';
              document.head.appendChild(link);
          }

          if (!window.L) {
              const script = document.createElement('script');
              script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
              script.integrity = 'sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=';
              script.crossOrigin = '';
              await new Promise(resolve => { script.onload = resolve; document.head.appendChild(script); });
          }

          if (window.L && mapRef.current && !mapRef.current._leaflet_id) {
              const mapInstance = window.L.map(mapRef.current, {
                  center: [52.2297, 21.0122], // Warsaw
                  zoom: 12,
              });
              window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              }).addTo(mapInstance);
              mapInstanceRef.current = mapInstance;
              setLeafletMap(mapInstance);
              const layerGroup = window.L.layerGroup().addTo(mapInstance);
              setMarkersLayer(layerGroup);
          }
      };
      initMap();
    }, []);

    const createBikeIcon = (station) => {
      if (!window.L || !station) return null;
      const bikes = typeof station.bikes === 'number' ? Math.max(0, station.bikes) : 0;
      const capacity = typeof station.capacity === 'number' && station.capacity > 0 ? station.capacity : 20;
      const fillRatio = capacity > 0 ? Math.min(1, bikes / capacity) : 0;
      let color = '#10b981'; // Green
      if (fillRatio < 0.2) color = '#ef4444'; // Red
      else if (fillRatio > 0.8) color = '#f59e0b'; // Orange
      const minSize = 16;
      const maxSize = 42;
      const maxCapacityForScaling = 50;

      const size = Math.round(
        minSize + (Math.min(capacity, maxCapacityForScaling) / maxCapacityForScaling) * (maxSize - minSize)
      );
      return window.L.divIcon({
          html: `<div style="width:${size}px; height:${size}px; background-color:${color}; border:2px solid white; border-radius:50%; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold; font-size:${size > 20 ? '12px' : '10px'}; box-shadow:0 2px 4px rgba(0,0,0,0.3); cursor:pointer;">${bikes}</div>`,
          className: 'custom-bike-icon',
          iconSize: [size, size],
          iconAnchor: [size / 2, size / 2]
      });
    };

    useEffect(() => {
      if (!leafletMap || !markersLayer || !window.L || !mapInstanceRef.current) return;
      
      console.log(`Updating map with ${safeStations.length} stations`);
      markersLayer.clearLayers();

      if (!safeStations.length) {
          return;
      }

      const stationMarkers = [];
      let validCoords = [];
      let invalidStations = 0;

      safeStations.forEach(station => {
          if (station && 
          typeof station.lat === 'number' && typeof station.lng === 'number' && 
          !isNaN(station.lat) && !isNaN(station.lng) &&
          station.lat >= -90 && station.lat <= 90 &&
          station.lng >= -180 && station.lng <= 180) {
              const icon = createBikeIcon(station);
              if (!icon) return;
              
              const popupContent = `
                <div style="min-width: 250px;">
                  <h4 style="margin: 0 0 8px 0; color: #1f2937; font-size: 16px;">${station.name || 'Station ' + station.id}</h4>
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px;">
                    <div><strong>ID:</strong> ${station.id}</div>
                    <div><strong>Bikes:</strong> ${station.bikes || 0}/${station.capacity || 'N/A'}</div>
                    <div><strong>Fill Rate:</strong> ${((station.bikes / station.capacity * 100) || 0).toFixed(0)}%</div>
                    <div><strong>Population (100m<sup>2</sup>):</strong> ${Math.round(station.population || 0)}</div>
                  </div>
                  ${station.poi_distances ? `
                    <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #e5e7eb;">
                      <div style="font-size: 11px; color: #6b7280;">
                        <div><strong>Distance to POIs:</strong></div>
                        <div>City Center: ${(station.poi_distances.city_center || 0).toFixed(1)}km</div>
                        <div>Metro: ${(station.poi_distances.metro || 0).toFixed(1)}km</div>
                        <div>University: ${(station.poi_distances.university || 0).toFixed(1)}km</div>
                        <div>Shopping: ${(station.poi_distances.mall || 0).toFixed(1)}km</div>
                      </div>
                    </div>
                  ` : ''}
                </div>
              `;
              
              const marker = window.L.marker([station.lat, station.lng], { icon })
                  .bindPopup(popupContent, { maxWidth: 300 })
                  .on('click', () => {
                    // Prevent map refresh on click
                    onStationClick(station);
                  });
              stationMarkers.push(marker);
              validCoords.push([station.lat, station.lng]);
          } else {
            invalidStations++;
            console.warn(`Invalid coordinates for station ${station?.id}:`, station);
          }
      });

      console.log(`Created ${stationMarkers.length} valid markers, ${invalidStations} invalid stations`);
      
      stationMarkers.forEach(m => markersLayer.addLayer(m));

      // Only fit bounds on initial load or significant change
      if (validCoords.length > 0 && !isTraining) {
          try {
              const currentBounds = leafletMap.getBounds();
              const newBounds = window.L.latLngBounds(validCoords);
              
              // Only fit if bounds changed significantly
              if (!currentBounds.contains(newBounds) || !newBounds.contains(currentBounds)) {
                  leafletMap.fitBounds(validCoords, { padding: [40, 40], maxZoom: 15 });
              }
          } catch (e) {
              console.error("Error fitting bounds:", e);
          }
      }

    }, [leafletMap, markersLayer, safeStations, onStationClick, isTraining]);

    return <div ref={mapRef} className="w-full h-[500px] md:h-[600px] rounded-lg shadow-lg border border-gray-200 relative" style={{ zIndex: 1, position: 'relative' }}/>
  };

  // Backend status check
  useEffect(() => {
    let isMounted = true;
    const checkBackendStatus = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/data_info');
        if (response.ok && isMounted) {
          const info = await response.json(); 
          setDataInfo(info);
          
          // Set optimal defaults from backend
          if (info.optimal_config) {
            setTrainingConfig(prev => ({
              ...prev,
              numStations: info.optimal_config.num_stations,
              maxBikes: info.optimal_config.max_bikes_users || 'all'
            }));
            
            if (info.optimal_config.training_period) {
              setTrainingDateRange({
                start: info.optimal_config.training_period.start,
                end: info.optimal_config.training_period.end
              });
            }
            
            if (info.optimal_config.reward_weights) {
              setRewardWeights(prev => ({
                stepPenalty: info.optimal_config.reward_weights.step_penalty || prev.stepPenalty,
                distErrorCoef: info.optimal_config.reward_weights.dist_error_coef || prev.distErrorCoef,
                moveCostCoef: info.optimal_config.reward_weights.move_cost_coef || prev.moveCostCoef,
                emptyStationPenalty: info.optimal_config.reward_weights.empty_station_penalty || prev.emptyStationPenalty,
                fullStationPenalty: info.optimal_config.reward_weights.full_station_penalty || prev.fullStationPenalty,
                tripFailurePenalty: info.optimal_config.reward_weights.trip_failure_penalty || prev.tripFailurePenalty,
                diversityBonus: info.optimal_config.reward_weights.diversity_bonus || prev.diversityBonus,
                successfulTripBonus: info.optimal_config.reward_weights.successful_trip_bonus || prev.successfulTripBonus,
                proactiveBonus: info.optimal_config.reward_weights.proactive_bonus || prev.proactiveBonus,
                distanceCostFactor: info.optimal_config.reward_weights.distance_cost_factor || prev.distanceCostFactor
              }));
            }
          }
          
          if (!info.loading_in_progress && info.static_data_loaded) setBackendLoading(false);
          else if (info.loading_in_progress) { setBackendLoading(true); setTimeout(checkBackendStatus, 2000); }
          else setBackendLoading(false);
        } else if (isMounted) {
           console.error('Backend status check failed:', response.status, response.statusText);
           setBackendLoading(false); 
           setDataInfo({ load_errors: [`Backend status check failed: ${response.status}`] });
        }
      } catch (error) { 
        if (isMounted) { 
          console.error('❌ Error checking backend status:', error); 
          setBackendLoading(false); 
          setDataInfo({ load_errors: ['Backend connection failed or server not responding.'] }); 
        } 
      }
    };
    checkBackendStatus();
    return () => { isMounted = false; };
  }, []);

  // Combine date and time
  useEffect(() => {
    if (selectedDate && selectedTime) {
        const fullDateTime = `${selectedDate} ${selectedTime}:00`;
        setSelectedDateTime(fullDateTime);
        setTrainingConfig(prevConfig => ({...prevConfig, selectedSnapshotTimestamp: fullDateTime}));
    }
  }, [selectedDate, selectedTime]);

  // Fetch available snapshots
  useEffect(() => {
    let isMounted = true;
    const fetchAvailableTimestamps = async () => {
      if (backendLoading || !dataInfo.static_data_loaded || !isMounted) return;
      try {
        const response = await fetch('http://localhost:5000/api/available_timestamps');
        if (response.ok && isMounted) {
          const timestampsData = await response.json();
          const formattedSnapshots = timestampsData.map(item => ({ timestamp: item.timestamp }));
          setAvailableSnapshots(formattedSnapshots);
          if (formattedSnapshots.length > 0 && !selectedDateTime) {
            // Default to test period if available
            const testPeriodTimestamp = formattedSnapshots.find(ts => ts.timestamp.startsWith('2023-11-16'));
            if (testPeriodTimestamp) {
              const [datePart, timePart] = testPeriodTimestamp.timestamp.split(' ');
              const [hours, minutes] = timePart.split(':');
              setSelectedDate(datePart); 
              setSelectedTime(`${hours}:${minutes}`);
            } else {
              const firstTimestamp = formattedSnapshots[0].timestamp;
              const [datePart, timePart] = firstTimestamp.split(' ');
              const [hours, minutes] = timePart.split(':');
              setSelectedDate(datePart); 
              setSelectedTime(`${hours}:${minutes}`);
            }
          }
        } else if (isMounted) {
            console.error("Failed to fetch available timestamps", response.status);
        }
      } catch (error) { 
        if (isMounted) console.error('❌ Error fetching available timestamps:', error); 
      }
    };
    fetchAvailableTimestamps();
    return () => { isMounted = false; };
  }, [backendLoading, dataInfo.static_data_loaded]);

  // Fetch weather data for specific timestamp
  const [currentWeather, setCurrentWeather] = useState(null);
  
  useEffect(() => {
    let isMounted = true;
    const fetchWeatherForTimestamp = async () => {
      if (!selectedDateTime || backendLoading || !dataInfo.static_data_loaded || !isMounted) return;
      try {
        const weatherResponse = await fetch('http://localhost:5000/api/weather');
        if (weatherResponse.ok && isMounted) { 
          const data = await weatherResponse.json(); 
          
          // Find closest weather record to selected timestamp
          if (data.length > 0) {
            const targetTime = new Date(selectedDateTime).getTime();
            let closestWeather = data[0];
            let minDiff = Math.abs(new Date(data[0].timestamp).getTime() - targetTime);
            
            for (let weather of data) {
              const diff = Math.abs(new Date(weather.timestamp).getTime() - targetTime);
              if (diff < minDiff) {
                minDiff = diff;
                closestWeather = weather;
              }
            }
            setCurrentWeather(closestWeather);
          }
        } else if (isMounted) { 
          console.error("Failed to fetch weather data", weatherResponse.status); 
        }
      } catch (error) { 
        if (isMounted) console.error('❌ Error fetching weather data:', error); 
      }
    };
    fetchWeatherForTimestamp();
    return () => { isMounted = false; };
  }, [selectedDateTime, backendLoading, dataInfo.static_data_loaded]);

  // FIXED: Fetch stations with training check
  useEffect(() => {
    let isMounted = true;
    const fetchStationsForSnapshot = async () => {
      if (!selectedDateTime || backendLoading || !dataInfo.static_data_loaded || !isMounted) {
        if (!selectedDateTime || !dataInfo.static_data_loaded) {
            setStations([]); 
            setSnapshotResponseInfo(null);
            setTimestampInfo(null);
        }
        return;
      }
      
      // Skip fetching during training
      if (isTraining) {
        console.log('Skipping station fetch during training');
        return;
      }
      
      try {
        console.log(`Fetching stations for timestamp: ${selectedDateTime}`);
        const stationsResponse = await fetch(`http://localhost:5000/api/stations?timestamp=${encodeURIComponent(selectedDateTime)}`);
        if (stationsResponse.ok && isMounted) {
          const data = await stationsResponse.json();
          console.log('Stations API response:', data);
          
          if (data.stations) { 
            setStations(data.stations);
            setCachedStations(data.stations); // Cache for use during training
            setSnapshotResponseInfo({ 
              requested_timestamp: data.requested_timestamp, 
              loaded_timestamp: data.loaded_timestamp, 
              message: data.message 
            });
            
            setTimestampInfo({
              requested: data.requested_timestamp,
              loaded: data.loaded_timestamp,
              is_exact_match: data.is_exact_match || false,
              message: data.message
            });
            
          } else {
            setStations(Array.isArray(data) ? data : []); 
            setCachedStations(Array.isArray(data) ? data : []);
            setSnapshotResponseInfo(data.message ? { message: data.message } : null);
          }
        } else if (isMounted) { 
            console.error("Failed to fetch stations for snapshot", stationsResponse.status);
            const errorData = await stationsResponse.json().catch(() => ({}));
            console.error("Error details:", errorData);
            setStations([]); 
            setSnapshotResponseInfo({ message: errorData.error || `Server error: ${stationsResponse.status}` }); 
            setTimestampInfo(null);
        }
      } catch (error) { 
        if (isMounted) { 
          console.error('❌ Error fetching stations for datetime:', error); 
          setStations([]); 
          setSnapshotResponseInfo({ message: `Connection error: ${error.message}` }); 
          setTimestampInfo(null);
        } 
      }
    };
    fetchStationsForSnapshot();
    return () => { isMounted = false; };
  }, [selectedDateTime, backendLoading, dataInfo.static_data_loaded, isTraining]);

  // IMPROVED: Enhanced training status monitoring
  useEffect(() => {
    let intervalId;
    if (isTraining) {
      setCanCancelTraining(true);
      intervalId = setInterval(async () => {
        try {
          const response = await fetch('http://localhost:5000/api/training/status');
          if (!response.ok) { 
            console.error('Error fetching training status:', response.status, response.statusText); 
            return; 
          }
          const status = await response.json();
          setTrainingProgress(status.progress || 0);
          
          if (status.logs && Array.isArray(status.logs)) {
            setTrainingLogs(prevLogs => {
              const existingLogsSet = new Set(prevLogs);
              const newLogs = status.logs.filter(log => !existingLogsSet.has(log));
              
              if (newLogs.length > 0) {
                const updatedLogs = [...prevLogs, ...newLogs].slice(-100);
                
                setTimeout(() => {
                  if (logsContainerRef.current) {
                    logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
                  }
                }, 100);
                
                return updatedLogs;
              }
              return prevLogs;
            });
          }
          
          if (status.metrics && typeof status.metrics === 'object') { 
            setTrainingMetrics(prev => ({
              timestep: status.metrics.timestep || prev.timestep || 0,
              episodes: status.metrics.episodes || prev.episodes || 0,
              ep_rew_mean: status.metrics.ep_rew_mean !== undefined ? status.metrics.ep_rew_mean : prev.ep_rew_mean,
              learning_rate: status.metrics.learning_rate || prev.learning_rate || 0.0003,
              loss: status.metrics.loss || prev.loss,
              elapsed_time: status.metrics.elapsed_time || prev.elapsed_time || 0
            })); 
          }
          
          // IMPROVED: Better training charts handling
          if (status.training_metrics && typeof status.training_metrics === 'object' && status.has_valid_metrics) {
            console.log('Received training metrics:', status.training_metrics);
            setTrainingCharts(status.training_metrics);
          }
          
          if (!status.isTraining) {
            clearInterval(intervalId);
            setIsTraining(false);
            setCanCancelTraining(false);
            if (status.results) {
              setTrainingResults(status.results);
              setModelPath(status.results.model_path || null);
              // Store training charts from results
              if (status.results.training_metrics) {
                setTrainingCharts(status.results.training_metrics);
              }
            } else {
              if (status.logs && status.logs.some(log => log.toLowerCase().includes("error") || log.toLowerCase().includes("failed"))) {
                  setTrainingProgress(0);
              }
              console.log("Training ended without full results. Status:", status.status, "Message:", status.message);
            }
          }
        } catch (error) { 
          console.error('Error fetching training status:', error); 
        }
      }, 2000);
    } else {
      setCanCancelTraining(false);
    }
    return () => { if (intervalId) clearInterval(intervalId); };
  }, [isTraining]);

  // Helper functions and components
  const LoadingScreen = () => (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center p-4">
        <Loader className="h-16 w-16 text-blue-600 mx-auto mb-6 animate-spin" />
        <h2 className="text-2xl font-semibold text-gray-800 mb-3">Loading Backend Data</h2>
        <p className="text-gray-600 mb-4 max-w-md mx-auto">
          {dataInfo.loading_in_progress 
            ? "Initializing station attributes, weather patterns, and scanning historical snapshots..." 
            : "Attempting to connect to the backend server..."}
        </p>
        {dataInfo.load_errors && dataInfo.load_errors.length > 0 && (
          <div className="mt-4 p-3 bg-red-50 rounded-lg border border-red-200 text-left max-w-md mx-auto">
            <div className="flex items-center text-red-700 font-medium mb-2">
              <AlertCircle className="h-5 w-5 mr-2 flex-shrink-0" />
              <span>Loading Issues Encountered:</span>
            </div>
            <ul className="text-xs text-red-600 space-y-1 list-disc list-inside">
              {dataInfo.load_errors.slice(0,3).map((error, index) => <li key={index}>{error}</li>)}
            </ul>
          </div>
        )}
      </div>
    </div>
  );

  // IMPROVED: Enhanced DataSelectionPanel moved under map with navigation
  const HistoricSnapshotsPanel = () => (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center">
          <Clock className="h-5 w-5 text-blue-600 mr-2" />
          Warsaw Bike-Sharing Stations Historic Snapshots
        </h3>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div> 
          <label className="block text-sm font-medium mb-1 text-gray-700">Date:</label> 
          <input 
            type="date" 
            value={selectedDate} 
            onChange={(e) => setSelectedDate(e.target.value)} 
            className="w-full border-gray-300 rounded-md shadow-sm text-sm p-2" 
          /> 
        </div>
        <div> 
          <label className="block text-sm font-medium mb-1 text-gray-700">Time:</label> 
          <input 
            type="time" 
            value={selectedTime} 
            onChange={(e) => setSelectedTime(e.target.value)} 
            className="w-full border-gray-300 rounded-md shadow-sm text-sm p-2" 
            step="300"
          /> 
        </div>
        <div className="grid grid-cols-2 gap-2 items-center">
          <div className="bg-gray-50 rounded p-3 text-center"> 
            <div className="font-bold text-blue-600">{dataInfo.stations_count || 0}</div> 
            <div className="text-gray-600 text-xs">Stations</div> 
          </div>
          <div className="bg-gray-50 rounded p-3 text-center"> 
            <div className="font-bold text-green-600">{stations.reduce((s, st) => s + (st.bikes || 0), 0)}</div> 
            <div className="text-gray-600 text-xs">Bikes</div> 
          </div>
        </div>
      </div>
      
      {selectedDateTime && (
        <div className="mt-3 p-2 bg-blue-100 rounded text-sm text-blue-700">
          Selected: {new Date(selectedDateTime).toLocaleString()}
        </div>
      )}
      
      {timestampInfo && (
        <div className="mt-2 space-y-1">
          {!timestampInfo.is_exact_match && (
            <div className="p-2 bg-yellow-100 rounded text-xs text-yellow-700 border border-yellow-200 flex items-center">
              <RefreshCw className="h-3 w-3 mr-1" />
              Using closest available time: {new Date(timestampInfo.loaded).toLocaleString()}
            </div>
          )}
          {timestampInfo.is_exact_match && (
            <div className="p-2 bg-green-100 rounded text-xs text-green-700 border border-green-200 flex items-center">
              <CheckCircle className="h-3 w-3 mr-1" />
              Exact timestamp match found
            </div>
          )}
        </div>
      )}
      
      {currentWeather && (
        <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center text-blue-800 text-sm mb-2">
            <Cloud className="h-4 w-4 mr-2" />
            <span className="font-medium">Weather Conditions</span>
          </div>
          <div className="grid grid-cols-3 gap-4 text-xs text-blue-700">
            <div>
              <div className="font-medium">Temperature</div>
              <div>{currentWeather.temperature?.toFixed(1)}°C</div>
            </div>
            <div>
              <div className="font-medium">Conditions</div>
              <div>{currentWeather.condition || 'Clear'}</div>
            </div>
            <div>
              <div className="font-medium">Wind Speed</div>
              <div>{(currentWeather.wind_speed || 0).toFixed(1)} km/h</div>
            </div>
          </div>
        </div>
      )}
      
      <div className="mt-3 bg-gray-50 rounded-lg p-3 border">
        <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2"> 
              <Database className="h-4 w-4 text-gray-500" /> 
              <span>Backend:</span> 
            </div>
            <div className={`flex items-center space-x-1 px-2 py-0.5 rounded-full text-xs ${dataInfo.loading_in_progress ? "bg-yellow-100 text-yellow-700" : dataInfo.static_data_loaded ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                {dataInfo.loading_in_progress ? <Loader className="h-3 w-3 animate-spin" /> : dataInfo.static_data_loaded ? <CheckCircle className="h-3 w-3" /> : <AlertCircle className="h-3 w-3" />}
                <span>{dataInfo.loading_in_progress ? "Loading..." : dataInfo.static_data_loaded ? "Ready" : "Error"}</span>
            </div>
        </div>
      </div>
    </div>
  );

  // Training functions
  const startTraining = async () => {
    if (!selectedDateTime || selectedDateTime.trim() === '') { 
      alert('Please select a date and time for context first.'); 
      return; 
    }
    if (!dataInfo.static_data_loaded) { 
      alert('Backend is not ready. Please wait for data to load.'); 
      return; 
    }
    if (!trainingDateRange.start || !trainingDateRange.end) { 
      alert('Please set both start and end dates for the training period.'); 
      return; 
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingResults(null);
    setTrainingLogs([`[${new Date().toLocaleTimeString()}] Training process initiated with custom reward weights...`]);
    setTrainingMetrics({ timestep: 0, episodes: 0, loss: null, learning_rate: null, ep_rew_mean: null, elapsed_time: 0 });
    setTrainingCharts(null);
    setModelPath(null);
    setShowComparison(false); // Hide any previous comparison

    const configPayload = {
      ...trainingConfig,
      trainingDateRange,
      selectedSnapshotTimestamp: selectedDateTime,
      rewardWeights: rewardWeights  // Include custom reward weights
    };
    console.log("Starting training with config payload:", configPayload);

    try {
      const response = await fetch('http://localhost:5000/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configPayload),
      });

      if (!response.ok) {
        const errorResult = await response.json().catch(() => ({ error: `Server error: ${response.status}` }));
        throw new Error(errorResult.error || `Training request failed with status ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Training successfully initiated by backend:', result.message);
      setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${result.message}`]);
    } catch (error) {
      console.error('❌ Error starting training:', error);
      alert(`Error starting training: ${error.message}`);
      setIsTraining(false);
      setCanCancelTraining(false);
      setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ERROR starting training: ${error.message}`]);
    }
  };

  const cancelTraining = async () => {
    if (!isTraining && !canCancelTraining) {
      console.log("No training in progress or cancellation not allowed yet.");
      return;
    }
    console.log("Attempting to cancel training...");
    setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Sending cancellation request...`]);
    try {
      const response = await fetch('http://localhost:5000/api/training/cancel', { method: 'POST' });
      if (response.ok) {
        const data = await response.json();
        alert(data.message || 'Training cancellation requested successfully.');
        setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${data.message || 'Cancellation request acknowledged by server.'}`]);
      } else {
        const errData = await response.json().catch(() => ({error: "Unknown error during cancellation."}));
        alert(`Failed to send cancel request: ${errData.error || response.statusText}`);
        setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ERROR sending cancellation: ${errData.error || response.statusText}`]);
      }
    } catch (error) {
      console.error('Error cancelling training:', error);
      alert('Error sending cancellation request to server.');
      setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Exception during cancellation: ${error.message}`]);
    }
  };

  const downloadModel = async () => {
    if (!modelPath) { 
      alert("Model path is not available for download."); 
      return; 
    }
    console.log("Attempting to download model from path:", modelPath);
    setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Preparing model download...`]);
    try {
      const response = await fetch(`http://localhost:5000/api/model/download?path=${encodeURIComponent(modelPath)}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none'; 
        a.href = url;
        const disposition = response.headers.get('content-disposition');
        let filename = `rl_bike_model_${new Date().toISOString().split('T')[0]}.zip`;
        if (disposition && disposition.includes('filename=')) {
            const filenameMatch = disposition.match(/filename="?([^"]+)"?/);
            if (filenameMatch && filenameMatch[1]) {
                filename = filenameMatch[1];
            }
        }
        a.download = filename;
        document.body.appendChild(a); 
        a.click(); 
        window.URL.revokeObjectURL(url); 
        a.remove();
        setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Model '${filename}' download started.`]);
      } else {
        const errData = await response.json().catch(() => ({error: "Failed to parse download error."}));
        alert(`Failed to download model: ${errData.error || `Server responded with ${response.status}`}`);
        setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ERROR downloading model: ${errData.error || response.status}`]);
      }
    } catch (error) {
      console.error('Error downloading model:', error);
      alert('An error occurred during the model download process.');
      setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Exception during model download: ${error.message}`]);
    }
  };

  const resetToOptimalWeights = () => {
    setRewardWeights({
      stepPenalty: -0.003,
      distErrorCoef: -0.12,
      moveCostCoef: -0.003,
      emptyStationPenalty: -0.25,
      fullStationPenalty: -0.25,
      tripFailurePenalty: -1.5,
      diversityBonus: 0.03,
      successfulTripBonus: 0.02,
      proactiveBonus: 0.01,
      distanceCostFactor: -0.002
    });
    setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Reset to optimal reward weights from research paper`]);
  };

  const RewardWeightsConfiguration = () => (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center">
          <Sliders className="h-5 w-5 mr-2 text-purple-600" />
          Reward Function Configuration
        </h3>
        <button 
          onClick={resetToOptimalWeights}
          className="text-sm bg-purple-100 text-purple-700 px-3 py-1 rounded hover:bg-purple-200 transition-colors"
          title="Reset to optimal weights from research paper"
        >
          Reset to Optimal
        </button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          { key: 'stepPenalty', label: 'Step Penalty', desc: 'Cost per time step' },
          { key: 'distErrorCoef', label: 'Distribution Error', desc: 'Penalty for bike imbalance' },
          { key: 'moveCostCoef', label: 'Move Cost', desc: 'Cost per bike moved' },
          { key: 'emptyStationPenalty', label: 'Empty Station Penalty', desc: 'Penalty for empty stations' },
          { key: 'fullStationPenalty', label: 'Full Station Penalty', desc: 'Penalty for full stations' },
          { key: 'tripFailurePenalty', label: 'Trip Failure Penalty', desc: 'Penalty for failed trips' },
          { key: 'diversityBonus', label: 'Diversity Bonus', desc: 'Reward for route diversity' },
          { key: 'successfulTripBonus', label: 'Success Bonus', desc: 'Reward for successful trips' },
          { key: 'proactiveBonus', label: 'Proactive Bonus', desc: 'Reward for demand anticipation' },
          { key: 'distanceCostFactor', label: 'Distance Cost', desc: 'Cost per km traveled' }
        ].map(({ key, label, desc }) => (
          <div key={key} className="space-y-1">
            <label className="block text-sm font-medium text-gray-700">
              {label}
              <span className="text-xs text-gray-500 block">{desc}</span>
            </label>
            <input
              type="number"
              step="0.001"
              value={rewardWeights[key]}
              onChange={(e) => setRewardWeights(prev => ({
                ...prev,
                [key]: parseFloat(e.target.value) || 0
              }))}
              className="w-full border-gray-300 rounded-md shadow-sm text-sm p-2 focus:ring-purple-500 focus:border-purple-500"
            />
          </div>
        ))}
      </div>
      
      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <h4 className="text-sm font-medium text-gray-800 mb-2">Current Configuration Summary:</h4>
        <div className="text-xs text-gray-600 grid grid-cols-2 gap-2">
          <div>• Penalties: {Object.entries(rewardWeights).filter(([k, v]) => v < 0).length} negative weights</div>
          <div>• Bonuses: {Object.entries(rewardWeights).filter(([k, v]) => v > 0).length} positive weights</div>
          <div>• Main penalty: {Math.abs(Math.min(...Object.values(rewardWeights))).toFixed(3)} (trip failures)</div>
          <div>• Main bonus: {Math.max(...Object.values(rewardWeights)).toFixed(3)} (diversity)</div>
        </div>
      </div>
    </div>
  );
  
  const TrainingConfiguration = () => (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        <Settings className="h-5 w-5 mr-2 text-blue-600" />
        Training Configuration
      </h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Number of Stations</label>
          <input 
            type="number" 
            value={trainingConfig.numStations} 
            onChange={(e) => setTrainingConfig({...trainingConfig, numStations: parseInt(e.target.value)})} 
            className="w-full border-gray-300 rounded-md shadow-sm text-sm p-2" 
            min="10" 
            max="310"
            step="10"
          />
          <div className="text-xs text-gray-500 mt-1">
            Top stations
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">
            Number of Bikes/Users
            <span className="text-xs text-gray-500 ml-1">- affects training quality</span>
          </label>
          <select 
            value={trainingConfig.maxBikes} 
            onChange={(e) => setTrainingConfig({...trainingConfig, maxBikes: e.target.value})} 
            className="w-full border-gray-300 rounded-md shadow-sm text-sm p-2"
          >
            <option value="all">All bikes (optimal)</option>
            <option value="2000">2000 bikes (high quality)</option>
            <option value="1000">1000 bikes (good quality)</option>
            <option value="500">500 bikes (fast training)</option>
          </select>
          <div className="text-xs text-gray-500 mt-1">
            Optimal: All bikes for best performance
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Training Steps</label>
          <input 
            type="number" 
            value={trainingConfig.steps} 
            onChange={(e) => setTrainingConfig({...trainingConfig, steps: parseInt(e.target.value)})} 
            className="w-full border-gray-300 rounded-md shadow-sm text-sm p-2" 
            min="10000" 
            step="10000"
          />
          <div className="text-xs text-gray-500 mt-1">
            Recommended: 100k+ steps for stable results
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Device</label>
          <select 
            value={trainingConfig.device} 
            onChange={(e) => setTrainingConfig({ ...trainingConfig, device: e.target.value })} 
            className="w-full border-gray-300 rounded-md px-3 py-2 text-sm shadow-sm"
          >
            <option value="auto">Auto-detect</option> 
            <option value="cuda">Force GPU (CUDA)</option> 
            <option value="cpu">Force CPU</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Random Seed</label>
          <input 
            type="number" 
            value={trainingConfig.seed} 
            onChange={(e) => setTrainingConfig({...trainingConfig, seed: parseInt(e.target.value) || 42})} 
            className="w-full border-gray-300 rounded-md shadow-sm text-sm p-2"
          />
        </div>
        
        <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
          <h4 className="text-sm font-medium text-blue-800 mb-2">Current Configuration:</h4>
          <div className="text-xs text-blue-700 space-y-1">
            <div>• {trainingConfig.numStations} stations will be analyzed</div>
            <div>• {trainingConfig.maxBikes === 'all' ? 'All' : trainingConfig.maxBikes} bikes will be used</div>
            <div>• Training for {trainingConfig.steps.toLocaleString()} steps</div>
            <div>• Training period: {trainingDateRange.start} to {trainingDateRange.end}</div>
            <div>• Test period: 2023-11-16 to 2023-11-22</div>
            <div>• Success metric: Trip completion rate optimization</div>
          </div>
        </div>
        
        <div>
          <button 
            onClick={startTraining} 
            disabled={isTraining || !selectedDateTime || !dataInfo.static_data_loaded} 
            className="w-full bg-blue-600 text-white px-4 py-2.5 rounded-md hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center font-medium"
          >
            {isTraining ? 
              <>
                <PauseCircle className="h-5 w-5 mr-2 animate-pulse" />
                Training... {Math.round(trainingProgress)}%
              </> : 
              <>
                <Play className="h-5 w-5 mr-2" />
                Start Training with Custom Rewards
              </>
            }
          </button>

          {isTraining && canCancelTraining && (
            <button 
              onClick={cancelTraining} 
              className="w-full mt-2 bg-red-600 text-white px-4 py-2.5 rounded-md hover:bg-red-700 flex items-center justify-center font-medium"
            >
              <X className="h-5 w-5 mr-2" />Cancel Training
            </button>
          )}
        </div>
        
        {isTraining && (
          <div className="mt-4 space-y-3">
            <div className="w-full bg-gray-200 rounded-full h-2.5"> 
              <div className="bg-blue-500 h-2.5 rounded-full transition-all duration-150 ease-linear" style={{ width: `${trainingProgress}%` }}></div> 
            </div>
            <div className="bg-gray-50 rounded p-3 text-xs space-y-1 border border-gray-200 shadow-sm">
              <div className="flex justify-between"><span className="text-gray-500">Timestep:</span><span className="font-mono text-gray-700">{trainingMetrics.timestep ? trainingMetrics.timestep.toLocaleString() : '0'}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Episodes:</span><span className="font-mono text-gray-700">{trainingMetrics.episodes || 0}</span></div>
              {trainingMetrics.ep_rew_mean !== null && trainingMetrics.ep_rew_mean !== undefined && (
              <div className="flex justify-between"><span className="text-gray-500">Avg Reward:</span><span className="font-mono text-gray-700">{typeof trainingMetrics.ep_rew_mean === 'number' ? trainingMetrics.ep_rew_mean.toFixed(2) : 'N/A'}</span></div>)}
              {trainingMetrics.loss !== null && typeof trainingMetrics.loss === 'number' && <div className="flex justify-between"><span className="text-gray-500">Loss:</span><span className="font-mono text-gray-700">{trainingMetrics.loss.toFixed(4)}</span></div>}
              <div className="flex justify-between"><span className="text-gray-500">Elapsed:</span><span className="font-mono text-gray-700">{(trainingMetrics.elapsed_time || 0).toFixed(0)}s</span></div>
            </div>
            {trainingLogs.length > 0 && (
              <div className="relative">
                <div className="flex items-center justify-between bg-gray-800 text-white px-3 py-2 rounded-t-md">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <span className="text-sm font-medium">Training Console</span>
                    <span className="text-xs text-gray-400">({trainingLogs.length}/100)</span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setTrainingLogs([])}
                      className="px-2 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-500 transition-colors"
                      title="Clear logs"
                    >
                      🗑️ Clear
                    </button>
                  </div>
                </div>
                
                <div 
                  ref={logsContainerRef}
                  className="bg-gray-900 text-green-400 rounded-b-md p-3 font-mono text-[11px] leading-relaxed max-h-48 overflow-y-auto border-2 border-gray-700 shadow-inner relative"
                >
                  {trainingLogs.slice().reverse().map((log, idx) => {
                    const isError = log.toLowerCase().includes('error') || log.toLowerCase().includes('failed');
                    const isSuccess = log.toLowerCase().includes('completed') || log.toLowerCase().includes('success');
                    const isWarning = log.toLowerCase().includes('warning') || log.toLowerCase().includes('warn');
                    const isProgress = log.includes('step') || log.includes('progress') || log.includes('%') || log.includes('Loading') || log.includes('Processing');
                    
                    const displayIndex = trainingLogs.length - idx;
                    
                    return (
                      <div 
                        key={`${trainingLogs.length}-${idx}`}
                        className={`whitespace-pre-wrap break-all mb-1 ${
                          isError ? 'text-red-400 font-semibold' : 
                          isSuccess ? 'text-green-300 font-medium' :
                          isWarning ? 'text-yellow-400' :
                          isProgress ? 'text-cyan-400' :
                          'text-green-400'
                        }`}
                      >
                        <span className="text-gray-500 text-[10px]">[{String(displayIndex).padStart(3, '0')}]</span> {log}
                        {idx < 3 && (
                          <span className="ml-2 px-1 py-0.5 bg-green-600 text-white text-[8px] rounded">NEW</span>
                        )}
                      </div>
                    );
                  })}
                  
                  {isTraining && (
                    <div className="absolute top-2 left-2 flex items-center space-x-1 bg-gray-800 px-2 py-1 rounded">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                      <span className="text-[10px] text-green-300">Live - Latest on top</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );

  // IMPROVED: Enhanced model comparison with same parameters
  const compareModels = async () => {
    if (!trainingResults?.model_path) {
      alert('No trained model available for comparison');
      return;
    }
    
    setIsComparing(true);
    setModelComparison(null);
    
    try {
      // Send the SAME configuration parameters used in training
      const comparisonConfig = {
        numStations: trainingConfig.numStations,
        maxBikes: trainingConfig.maxBikes,
        testPeriod: {
          start: '2023-11-16', // Test period after training
          end: '2023-11-22'
        }
      };
      
      console.log('Comparing models with config:', comparisonConfig);
      
      const response = await fetch('http://localhost:5000/api/training/compare_models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(comparisonConfig),
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Comparison results:', data);
        setModelComparison(data.comparison);
        setShowComparison(true);
      } else {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        console.error('Comparison failed:', errorData);
        alert(`Failed to compare models: ${errorData.error}`);
      }
    } catch (error) {
      console.error('Error comparing models:', error);
      alert('Error comparing models. Check console for details.');
    } finally {
      setIsComparing(false);
    }
  };

  // FIXED: Enhanced training charts with better distribution error handling
  const TrainingChartsDisplay = () => {
    if (!trainingCharts || !trainingCharts.timesteps || trainingCharts.timesteps.length === 0) {
      return (
        <div className="mt-6 bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 mr-2 text-green-600" />
            Training Progress Charts
          </h3>
          <div className="text-center text-gray-500 py-8">
            <Loader className="h-8 w-8 mx-auto mb-3 animate-spin" />
            <p>Training metrics will appear here during training...</p>
          </div>
        </div>
      );
    }

    return (
      <div className="mt-6 bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <TrendingUp className="h-5 w-5 mr-2 text-green-600" />
          Training Progress Charts
          <span className="ml-3 text-sm text-gray-500">({trainingCharts.timesteps.length} data points)</span>
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Success Rate Chart */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-medium text-gray-800 mb-4 text-center">Trip Success Rate Over Time</h4>
            <div className="h-32 flex items-end space-x-1 justify-center">
              {trainingCharts.trip_success_rate && trainingCharts.trip_success_rate.slice(-20).map((rate, idx) => (
                <div
                  key={idx}
                  className="bg-green-500 w-2 rounded-t"
                  style={{ height: `${Math.max(rate * 1.2, 5)}%` }}
                  title={`${rate.toFixed(1)}%`}
                />
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-2 text-center">
              Latest: {trainingCharts.trip_success_rate?.[trainingCharts.trip_success_rate.length - 1]?.toFixed(1) || 'N/A'}%
            </div>
          </div>

          {/* Reward Chart */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-medium text-gray-800 mb-4 text-center">Average Reward</h4>
            <div className="h-32 flex items-end space-x-1 justify-center">
              {trainingCharts.rewards && trainingCharts.rewards.slice(-20).map((reward, idx) => {
                const normalizedHeight = Math.max((reward + 20) * 2, 5); // Normalize negative rewards for display
                return (
                  <div
                    key={idx}
                    className={`w-2 rounded-t ${reward >= 0 ? 'bg-blue-500' : 'bg-red-400'}`}
                    style={{ height: `${normalizedHeight}%` }}
                    title={`${reward.toFixed(2)}`}
                  />
                );
              })}
            </div>
            <div className="text-xs text-gray-500 mt-2 text-center">
              Latest: {trainingCharts.rewards?.[trainingCharts.rewards.length - 1]?.toFixed(2) || 'N/A'}
            </div>
          </div>

          {/* Bikes Moved Chart */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-medium text-gray-800 mb-4 text-center">Bikes Moved Per Episode</h4>
            <div className="h-32 flex items-end space-x-1 justify-center">
              {trainingCharts.bikes_moved && trainingCharts.bikes_moved.slice(-20).map((moved, idx) => (
                <div
                  key={idx}
                  className="bg-purple-500 w-2 rounded-t"
                  style={{ height: `${Math.max(moved, 5)}%` }}
                  title={`${moved} bikes`}
                />
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-2 text-center">
              Latest: {trainingCharts.bikes_moved?.[trainingCharts.bikes_moved.length - 1] || 'N/A'} bikes
            </div>
          </div>

          {/* Distribution Error Chart - FIXED to show actual values */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-medium text-gray-800 mb-4 text-center">Distribution Error</h4>
            <div className="h-32 flex items-end space-x-1 justify-center">
              {trainingCharts.distribution_error && trainingCharts.distribution_error.slice(-20).map((error, idx) => {
                const errorData = trainingCharts.distribution_error.slice(-20);
                const maxError = Math.max(...errorData.filter(e => e !== null && e !== undefined));
                
                // If error is null/undefined, skip this bar
                if (error === null || error === undefined) {
                  return <div key={idx} className="w-2" />;
                }
                
                let height;
                if (maxError === 0 || maxError < 0.0001) {
                  // All values are very small
                  height = error === 0 ? 10 : 20;
                } else {
                  // Scale relative to max, ensure minimum visibility
                  height = Math.max((error / maxError) * 80, 5);
                }
                
                // Color coding based on error level
                let colorClass = 'bg-orange-500';
                if (error === 0) colorClass = 'bg-green-400';
                else if (error < 0.01) colorClass = 'bg-yellow-400';
                else if (error < 0.05) colorClass = 'bg-orange-400';
                else colorClass = 'bg-red-500';
                
                return (
                  <div
                    key={idx}
                    className={`${colorClass} w-2 rounded-t transition-all duration-200`}
                    style={{ height: `${height}%` }}
                    title={`Error: ${error.toFixed(6)}`}
                  />
                );
              })}
            </div>
            <div className="text-xs text-gray-500 mt-2 text-center">
              Latest: {(() => {
                const latestError = trainingCharts.distribution_error?.[trainingCharts.distribution_error.length - 1];
                if (latestError === null || latestError === undefined) return 'N/A';
                return latestError === 0 ? '0.000000' : latestError.toFixed(6);
              })()}
              {trainingCharts.distribution_error?.[trainingCharts.distribution_error.length - 1] === 0 && 
                <span className="ml-1 text-green-600 font-medium">✓ Perfect Balance</span>
              }
            </div>
          </div>
        </div>
      </div>
    );
  };

  // IMPROVED: Enhanced model comparison display
  const ModelComparison = () => {
    if (!modelComparison) return null;

    const models = ['baseline', 'user'];
    const modelData = models.map(key => modelComparison[key]).filter(Boolean);

    return (
      <div className="mt-8 bg-white rounded-lg shadow-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold flex items-center text-gray-800">
            <BarChart3 className="h-6 w-6 mr-2 text-purple-600" />
            Baseline vs Your Model Comparison
            <span className="ml-3 text-sm text-gray-500">(Same Parameters)</span>
          </h3>
          <button 
            onClick={() => setShowComparison(false)}
            className="text-gray-500 hover:text-gray-700"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Fair Comparison Notice */}
        <div className="mb-6 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center text-blue-800 text-sm">
            <CheckCircle className="h-4 w-4 mr-2" />
            <span className="font-medium">Fair Comparison:</span>
            <span className="ml-1">Both models tested with identical parameters and data</span>
          </div>
        </div>

        {/* Comparison Table */}
        <div className="overflow-x-auto mb-8">
          <table className="w-full border-collapse border border-gray-200 rounded-lg">
            <thead>
              <tr className="bg-gray-50">
                <th className="border border-gray-200 px-4 py-3 text-left font-semibold text-gray-700">Model</th>
                <th className="border border-gray-200 px-4 py-3 text-center font-semibold text-gray-700">Success Rate ⭐</th>
                <th className="border border-gray-200 px-4 py-3 text-center font-semibold text-gray-700">Avg Reward</th>
                <th className="border border-gray-200 px-4 py-3 text-center font-semibold text-gray-700">Total Relocations</th>
                <th className="border border-gray-200 px-4 py-3 text-center font-semibold text-gray-700">Avg Distance</th>
              </tr>
            </thead>
            <tbody>
              {modelData.map((model, idx) => {
                const isUser = model.name.includes('Your');
                const isBest = modelData.length > 1 && model.success_rate === Math.max(...modelData.map(m => m.success_rate));
                
                return (
                  <tr key={idx} className={`${isUser ? 'bg-blue-50' : ''} ${isBest ? 'ring-2 ring-green-300' : ''}`}>
                    <td className="border border-gray-200 px-4 py-3">
                      <div>
                        <div className="font-medium text-gray-900">{model.name}</div>
                        <div className="text-sm text-gray-500">{model.description}</div>
                      </div>
                    </td>
                    <td className="border border-gray-200 px-4 py-3 text-center">
                      <span className={`font-mono text-lg ${model.success_rate > 50 ? 'text-green-600' : model.success_rate > 20 ? 'text-yellow-600' : 'text-red-600'}`}>
                        {model.success_rate.toFixed(1)}%
                      </span>
                      {isBest && <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">BEST</span>}
                    </td>
                    <td className="border border-gray-200 px-4 py-3 text-center">
                      <span className={`font-mono ${model.avg_reward > -20 ? 'text-green-600' : 'text-red-600'}`}>
                        {model.avg_reward.toFixed(1)}
                      </span>
                    </td>
                    <td className="border border-gray-200 px-4 py-3 text-center font-mono text-gray-700">
                      {model.total_relocations}
                    </td>
                    <td className="border border-gray-200 px-4 py-3 text-center">
                      <span className={`font-mono text-sm ${
                        isUser ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {model.avg_distance?.toFixed(2) || 'N/A'} km
                      </span>
                      <div className="text-xs text-gray-500 mt-1">
                        {isUser ? 'Optimized' : 'Random'}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Performance Analysis */}
        <div className="space-y-6">
          <h4 className="text-lg font-semibold text-gray-800 flex items-center">
            <MapPin className="h-5 w-5 mr-2 text-indigo-600" />
            Performance Analysis
          </h4>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {modelData.map((model, idx) => (
              <div key={idx} className="bg-gray-50 rounded-lg p-4 border">
                <h5 className="font-medium text-gray-800 mb-3 text-center">{model.name}</h5>
                
                <div className="bg-white rounded border h-64 p-3 overflow-y-auto">
                  <div className="text-xs text-gray-600 mb-2">Recent Decisions:</div>
                  {model.decisions && model.decisions.length > 0 ? (
                    <div className="space-y-2">
                      {model.decisions.slice(0, 8).map((decision, didx) => (
                        <div key={didx} className="flex items-center justify-between text-xs bg-gray-50 p-2 rounded">
                          <div className="flex-1">
                            <div className="font-mono text-gray-800">
                              {decision.from_station} → {decision.to_station}
                            </div>
                            <div className="text-gray-600">
                              {decision.bikes_moved} bikes, {decision.distance?.toFixed(1)}km
                            </div>
                          </div>
                          <div className={`px-2 py-1 rounded text-xs ${
                            decision.efficiency === 'Random' ? 'bg-red-100 text-red-700' :
                            'bg-blue-100 text-blue-700'
                          }`}>
                            {decision.efficiency}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-gray-500 text-center py-8">No decisions recorded</div>
                  )}
                </div>
                
                <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-white p-2 rounded text-center">
                    <div className="text-gray-500">Success Rate</div>
                    <div className="font-bold text-gray-700">{model.success_rate.toFixed(1)}%</div>
                  </div>
                  <div className="bg-white p-2 rounded text-center">
                    <div className="text-gray-500">Relocations</div>
                    <div className="font-bold text-gray-700">{model.total_relocations}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Results Summary */}
        <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h5 className="font-medium text-blue-800 mb-2">Test Results Summary (Nov 16-22, 2023):</h5>
          <div className="text-sm text-blue-700 space-y-1">
            {(() => {
              if (modelData.length < 2) return <div>• Not enough models to compare</div>;
              
              const userModel = modelData.find(m => m.name.includes('Your'));
              const baselineModel = modelData.find(m => m.name.includes('Baseline'));
              
              const insights = [];
              
              if (userModel && baselineModel) {
                const successImprovement = userModel.success_rate - baselineModel.success_rate;
                const rewardImprovement = userModel.avg_reward - baselineModel.avg_reward;
                
                insights.push(`✅ Success rate: ${userModel.success_rate.toFixed(1)}% vs ${baselineModel.success_rate.toFixed(1)}% baseline (${successImprovement > 0 ? '+' : ''}${successImprovement.toFixed(1)}pp)`);
                insights.push(`📈 Reward improvement: ${rewardImprovement > 0 ? '+' : ''}${rewardImprovement.toFixed(1)} points`);
                
                if (successImprovement > 15) {
                  insights.push(`🎯 Excellent performance! Your model significantly outperforms random decisions`);
                } else if (successImprovement > 5) {
                  insights.push(`👍 Good improvement over baseline, your model learned effective strategies`);
                } else if (successImprovement > 0) {
                  insights.push(`📊 Modest improvement - consider longer training or tuning reward weights`);
                } else {
                  insights.push(`⚠️ Model underperforming baseline - try different reward weights or more training steps`);
                }
              }
              
              const bestModel = modelData.reduce((best, current) => 
                current.success_rate > best.success_rate ? current : best
              );
              insights.push(`🏆 Best performing model: ${bestModel.name} (${bestModel.success_rate.toFixed(1)}% success rate)`);
              
              return insights.map((insight, i) => <div key={i}>• {insight}</div>);
            })()}
          </div>
        </div>
      </div>
    );
  };
  
  const TrainingResults = () => {
    if (!trainingResults) return null;
    const trainingTimeMinutes = trainingResults.training_time_seconds != null ? (trainingResults.training_time_seconds / 60).toFixed(1) : 'N/A';
    
    // Focus on key metrics with success rate as primary
    const displayMetrics = [
        { label: "Success Rate", value: trainingResults.success_rate, unit: '%', color: "green", primary: true },
        { label: "Avg Reward", value: trainingResults.avg_evaluation_reward, unit: '', color: "blue" },
        { label: "Total Relocations", value: trainingResults.total_relocations, unit: '', color: "purple" },
        { label: "Training Time", value: trainingTimeMinutes, unit: 'min', color: "gray" },
    ];

    return (
      <div className="mt-6 bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold flex items-center text-gray-800"> 
            <BarChart3 className="h-5 w-5 mr-2 text-green-600" /> Training Results 
          </h3>
          <div className="flex space-x-2">
            <button 
              onClick={() => setShowResults(!showResults)} 
              className="bg-gray-100 px-3 py-1.5 rounded-md text-sm hover:bg-gray-200 flex items-center shadow-sm"
            > 
              <Zap className="h-4 w-4 mr-1" /> {showResults ? 'Hide' : 'Show'} Details 
            </button>
            <button 
              onClick={compareModels} 
              disabled={isComparing}
              className="bg-purple-600 text-white px-3 py-1.5 rounded-md text-sm hover:bg-purple-700 flex items-center shadow-sm disabled:opacity-50"
            > 
              <BarChart3 className="h-4 w-4 mr-1" /> 
              {isComparing ? 'Comparing...' : 'Compare vs Baseline'}
            </button>
            {modelPath && ( 
              <button 
                onClick={downloadModel} 
                className="bg-green-600 text-white px-3 py-1.5 rounded-md text-sm hover:bg-green-700 flex items-center shadow-sm"
              > 
                <Download className="h-4 w-4 mr-1" /> Download Model 
              </button> 
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 text-center">
          {displayMetrics.map(metric => (
            (metric.value !== null && metric.value !== undefined && metric.value !== "N/A") && 
            <div key={metric.label} className={`bg-${metric.color}-50 p-4 rounded-lg border border-${metric.color}-200 shadow ${metric.primary ? 'ring-2 ring-green-300' : ''}`}>
              <div className={`text-3xl font-bold text-${metric.color}-600 ${metric.primary ? 'text-4xl' : ''}`}>
                {typeof metric.value === 'number' ? metric.value.toFixed(1) : metric.value}
                {metric.unit && <span className="text-lg ml-1">{metric.unit}</span>}
              </div>
              <div className={`text-sm text-gray-600 mt-1 ${metric.primary ? 'font-medium' : ''}`}>{metric.label}</div>
              {metric.primary && <div className="text-xs text-green-600 mt-1">Primary Metric</div>}
            </div>
          ))}
        </div>

        {modelPath && (
          <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200 shadow-sm">
            <div className="flex items-center justify-between">
              <div> 
                <div className="font-medium text-blue-800">Trained Model Ready</div> 
                <div className="text-sm text-blue-700 truncate"> 
                  Path: <code className="bg-blue-100 px-1 py-0.5 rounded text-xs">{modelPath}</code> 
                </div> 
              </div>
              <Zap className="h-6 w-6 text-blue-500" />
            </div>
          </div>
        )}
        
        {/* Training Charts */}
        <TrainingChartsDisplay />
        
        {/* Model Comparison */}
        {showComparison && <ModelComparison />}
      </div>
    );
  };

  if (backendLoading || (!dataInfo.static_data_loaded && dataInfo.loading_in_progress)) {
    return <LoadingScreen />;
  }

  // Use cached stations during training to prevent map refresh
  const displayStations = isTraining ? cachedStations : stations;

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800">
      <header className="bg-white shadow-md sticky top-0 z-50 relative">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <img 
                src="https://i.ibb.co/HL182sBs/e36fa2bb-b33b-4491-b034-93130804203f.png" 
                alt="Company Logo" 
                className="w-20 h-20 object-contain"
                onError={(e) => {
                  // Fallback to Brain icon if logo fails to load
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div> 
                <h1 className="text-2xl font-bold text-gray-900">Bike-sharing Rebalancing Platform - RL Agent For Optimization</h1> 
                <p className="text-sm text-gray-500">Optimize your BSS Using Custom Reward Functions</p> 
              </div>
            </div>
            
            {/* Add debug info in header */}
            <div className="text-xs text-gray-500">
              Backend: {dataInfo.static_data_loaded ? '✅' : '❌'} | 
              Stations: {displayStations.length} | 
              Training: {isTraining ? '🔄' : '⏸️'}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <section className="lg:col-span-2 space-y-8">
            <div className="bg-white rounded-lg shadow-xl p-4 sm:p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold flex items-center"> 
                  <MapPin className="h-6 w-6 mr-2 text-blue-500" /> Warsaw Bike-Sharing Stations (Snapshot) 
                </h2>
                <div className="text-sm text-gray-500"> 
                  {displayStations.length > 0 ? `${displayStations.length} stations` : "Loading stations..."} 
                  {selectedDateTime && ` | ${new Date(selectedDateTime).toLocaleString()}`}
                </div>
              </div>
              <LeafletWarsawMap stations={displayStations} selectedStation={selectedStation} onStationClick={setSelectedStation} />
            </div>
            
            <HistoricSnapshotsPanel />
          </section>
          
          <aside className="space-y-8">
            <div className="bg-white rounded-lg shadow-xl p-6">
              <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2"> 
                    <Database className="h-4 w-4 text-gray-500" /> 
                    <span>System Status:</span> 
                  </div>
                  <div className={`flex items-center space-x-1 px-2 py-0.5 rounded-full text-xs ${dataInfo.loading_in_progress ? "bg-yellow-100 text-yellow-700" : dataInfo.static_data_loaded ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                      {dataInfo.loading_in_progress ? <Loader className="h-3 w-3 animate-spin" /> : dataInfo.static_data_loaded ? <CheckCircle className="h-3 w-3" /> : <AlertCircle className="h-3 w-3" />}
                      <span>{dataInfo.loading_in_progress ? "Loading..." : dataInfo.static_data_loaded ? "Ready" : "Error"}</span>
                  </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-xl p-6">
              <div className="flex items-center mb-4"> 
                <CalendarDays className="h-5 w-5 text-green-600 mr-2" /> 
                <h3 className="font-semibold text-gray-800 text-lg">Training Period</h3> 
              </div>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Start Date:</label>
                  <input 
                    type="date" 
                    value={trainingDateRange.start} 
                    onChange={(e) => setTrainingDateRange({...trainingDateRange, start: e.target.value})} 
                    className="w-full border-gray-300 rounded-md text-sm p-2 shadow-sm focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">End Date:</label>
                  <input 
                    type="date" 
                    value={trainingDateRange.end} 
                    onChange={(e) => setTrainingDateRange({...trainingDateRange, end: e.target.value})} 
                    className="w-full border-gray-300 rounded-md text-sm p-2 shadow-sm focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>
              <div className="mt-3 p-2 bg-green-100 rounded text-sm text-green-700">
                <strong>Optimal:</strong> May 6 9AM - Nov 29 3PM, 2023 (Maximal range)
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-xl p-6">
              <div className="flex items-center mb-4"> 
                <BarChart3 className="h-5 w-5 text-purple-600 mr-2" /> 
                <h3 className="font-semibold text-gray-800 text-lg">Testing Period</h3> 
              </div>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Start Date:</label>
                  <input 
                    type="date" 
                    value={testingDateRange.start} 
                    onChange={(e) => setTestingDateRange({...testingDateRange, start: e.target.value})} 
                    className="w-full border-gray-300 rounded-md text-sm p-2 shadow-sm focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">End Date:</label>
                  <input 
                    type="date" 
                    value={testingDateRange.end} 
                    onChange={(e) => setTestingDateRange({...testingDateRange, end: e.target.value})} 
                    className="w-full border-gray-300 rounded-md text-sm p-2 shadow-sm focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
              </div>
              <div className="mt-3 p-2 bg-purple-100 rounded text-sm text-purple-700">
                <strong>Default:</strong> Nov 16 - Nov 22, 2023 (Test period after training)
              </div>
            </div>
            
            <RewardWeightsConfiguration />
            <TrainingConfiguration />
          </aside>
        </div>
        {trainingResults && <TrainingResults />}
      </main>
    </div>
  );
};

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) { 
    super(props); 
    this.state = { hasError: false, error: null, errorInfo: null }; 
  }
  
  static getDerivedStateFromError(error) { 
    return { hasError: true }; 
  }
  
  componentDidCatch(error, errorInfo) { 
    console.error('❌ React Error Boundary:', error, errorInfo); 
    this.setState({ error, errorInfo }); 
  }
  
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-red-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-lg p-6 max-w-2xl w-full">
            <div className="flex items-center mb-4">
              <AlertCircle className="h-8 w-8 text-red-500 mr-3" />
              <h1 className="text-xl font-bold text-red-800">Application Error</h1>
            </div>
            <p className="text-gray-700 mb-4">An error occurred in the application. Please check the console or try reloading.</p>
            <details className="bg-gray-50 rounded p-4 text-sm">
              <summary className="cursor-pointer font-medium text-gray-800 mb-2">Error Details</summary>
              <pre className="whitespace-pre-wrap text-red-600">{this.state.error?.toString()}</pre>
              <pre className="whitespace-pre-wrap text-gray-600 mt-2">{this.state.errorInfo?.componentStack}</pre>
            </details>
            <button 
              onClick={() => window.location.reload()} 
              className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              Reload Application
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const BikeRebalancingAppWithErrorBoundary = () => ( 
  <ErrorBoundary> 
    <BikeRebalancingApp /> 
  </ErrorBoundary> 
);

export default BikeRebalancingAppWithErrorBoundary;