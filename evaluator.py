#evaluator.py
"""
Module for evaluating the accuracy of dead reckoning predictions against actual flight data.
"""
import json
import statistics
import bisect
import os
from collections import defaultdict
from typing import List
from config_loader import CONFIG, LOG_DIR
from coord_utils import distance_km

def find_closest_actual(timestamps: List[float], target_time: float) -> int:
    """Finds the index of the closest timestamp in a sorted list using bisection."""
    if not timestamps:
        return -1
    pos = bisect.bisect_left(timestamps, target_time)
    if pos == 0:
        return 0
    if pos == len(timestamps):
        return len(timestamps) - 1
    before, after = timestamps[pos - 1], timestamps[pos]
    return pos if abs(after - target_time) < abs(before - target_time) else pos - 1

def calculate_errors() -> dict:
    """Computes prediction errors by comparing prediction logs to actual flight data logs."""
    actuals_file = os.path.join(LOG_DIR, 'actuals.json')
    predictions_file = os.path.join(LOG_DIR, 'predictions.json')
    
    try:
        with open(actuals_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            # Handle both the old list format and the new {"metadata": ..., "data": [...]} format
            actuals = loaded.get('data', loaded if isinstance(loaded, list) else [])
        with open(predictions_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            predictions = loaded.get('data', loaded if isinstance(loaded, list) else [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing log files: {e}")
        return {}

    actuals_by_icao = defaultdict(list)
    timestamps_by_icao = {}
    for ac in actuals:
        icao = ac.get('icao')
        ts = ac.get('timestamp')
        if icao is None or ts is None:
            continue
        actuals_by_icao[icao].append(ac)

    # Pre-sort actuals and extract timestamps for faster lookup
    for icao, data in actuals_by_icao.items():
        data.sort(key=lambda x: x['timestamp'])
        timestamps_by_icao[icao] = [x['timestamp'] for x in data]
    
    errors_by_delta = defaultdict(list)

    for pred in predictions:
        icao = pred.get('icao')
        # Handle the structure of predictions.json, assuming it contains prediction dicts
        preds = pred.get('predictions') # Assuming predictions are nested under this key
        if not icao or icao not in actuals_by_icao or not isinstance(preds, dict):
            continue
        
        actual_list = actuals_by_icao[icao]
        ts_list = timestamps_by_icao[icao]
        
        for delta_key, est in preds.items():
            est_time = est.get('est_time')
            est_lat = est.get('est_lat')
            est_lon = est.get('est_lon')
            if est_time is None or est_lat is None or est_lon is None:
                continue

            idx = find_closest_actual(ts_list, est_time)
            if idx == -1:
                continue
            
            closest_actual = actual_list[idx]
            
            # Extract the numeric delta value from the key (e.g., 'delta_30')
            try:
                # Ensure the key format is expected and extract the number part
                if not delta_key.startswith("delta_"):
                    continue
                delta_val = float(delta_key.split('_', 1)[1])
            except (ValueError, IndexError, TypeError):
                # Skip if key format is wrong or value isn't numeric
                continue

            # Check if the closest actual timestamp is within a reasonable tolerance
            # Tolerance increases slightly with prediction horizon
            tolerance = 2.0 + (delta_val / 30.0) * 2.0 # Example: up to 4s tolerance for 30s prediction
            if abs(closest_actual['timestamp'] - est_time) <= tolerance:
                lat = closest_actual.get('lat')
                lon = closest_actual.get('lon')
                if lat is None or lon is None:
                    continue
                
                dist_km = distance_km(est_lat, est_lon, lat, lon)
                dist_nm = dist_km / 1.852 # Convert km to nm
                errors_by_delta[delta_key].append(dist_nm) # Append nm error
    
    # Calculate statistics for each delta
    stats = {}
    for delta_key, errs in errors_by_delta.items():
        if errs:
            stats[delta_key] = {
                'mean': statistics.mean(errs),
                'median': statistics.median(errs),
                'max': max(errs),
                'count': len(errs)
            }
        else:
            # Handle cases where no valid errors were found for a delta
            stats[delta_key] = {'mean': 0, 'median': 0, 'max': 0, 'count': 0}
    
    return stats

if __name__ == "__main__":
    error_stats = calculate_errors()
    if error_stats:
        print("Prediction Error Evaluation (in Nautical Miles):")
        # Sort the output numerically by the delta value for better readability
        try:
            sorted_items = sorted(
                error_stats.items(),
                # Safely extract numeric part of delta_key, default to large number if format is unexpected
                key=lambda kv: float(kv[0].split('_')[1]) if kv[0].startswith("delta_") and len(kv[0].split('_')) > 1 else float('inf')
            )
        except ValueError:
            # Fallback to simple string sort if key parsing fails
            sorted_items = sorted(error_stats.items())

        for delta, s in sorted_items:
            # Format output nicely
            print(f"  {delta:>9}: Mean={s['mean']:.4f}, Median={s['median']:.4f}, Max={s['max']:.4f}, Count={s['count']}")
    else:
        print("No valid prediction data found or logs missing to evaluate.")