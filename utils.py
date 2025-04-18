# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def parse_timestamp(time_str):
    try:
        # For timestamps in format "20:32.6" (likely from your data)
        if ':' in time_str and '.' in time_str and len(time_str.split(':')) == 2:
            # Split into minutes and seconds
            minutes, seconds = time_str.split(':')
            # Create a timedelta object
            total_seconds = float(minutes) * 60 + float(seconds)
            # Return the number of seconds rather than a datetime
            return total_seconds
        elif ':' in time_str:
            # Standard timestamp with date
            return pd.to_datetime(time_str)
        else:
            # Try standard format as fallback
            return pd.to_datetime(time_str)
    except Exception as e:
        print(f"Parse error for '{time_str}': {e}")
        return None

def calculate_statistics(data):
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
    }

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def filter_outliers(data, lower_percentile=1, upper_percentile=99):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    print(f"Filtered {len(data) - len(filtered_data)} outliers "
          f"({(len(data) - len(filtered_data))/len(data)*100:.2f}%)")
    
    return filtered_data

def plot_histogram(data, bins=50, title=None, xlabel=None, ylabel=None, 
                 figsize=(10, 6), log_scale=False, save_path=None):
    plt.figure(figsize=figsize)
    sns.histplot(data, bins=bins, kde=True)
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return plt.gcf()

