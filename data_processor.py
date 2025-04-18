# data_processor.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob
import argparse
from datetime import datetime
from utils import load_csv, parse_timestamp, calculate_statistics, create_directory, plot_histogram, filter_outliers

class PacketDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_metaverse_data(self, app_name=None, max_files=None):
        metaverse_dir = os.path.join(self.data_dir, 'Metaverse')
        
        if app_name is not None:
            app_dirs = [os.path.join(metaverse_dir, app_name)]
        else:
            app_dirs = [os.path.join(metaverse_dir, d) for d in os.listdir(metaverse_dir) 
                       if os.path.isdir(os.path.join(metaverse_dir, d))]
        
        all_dfs = []
        
        for app_dir in app_dirs:
            app_name = os.path.basename(app_dir)
            file_pattern = os.path.join(app_dir, f"{app_name}_*.csv")
            csv_files = sorted(glob.glob(file_pattern))
            
            if max_files is not None:
                csv_files = csv_files[:max_files]
            
            for file_path in csv_files:
                df = load_csv(file_path)
                if df is not None:
                    # Add metadata columns
                    df['app_name'] = app_name
                    df['file_name'] = os.path.basename(file_path)
                    all_dfs.append(df)
                    print(f"Loaded {file_path}, shape: {df.shape}")
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Combined shape: {combined_df.shape}")
            return combined_df
        else:
            print("No data loaded!")
            return None
    
    def parse_timestamps(self, df):
        """
        Parse the 'Time' column to proper datetime format.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'Time' column
            
        Returns:
            pd.DataFrame: DataFrame with parsed timestamps
        """
        df = df.copy()
        
        # Parse timestamps
        df['timestamp'] = df['Time'].apply(parse_timestamp)
        
        # Filter out rows with invalid timestamps
        n_before = len(df)
        df = df.dropna(subset=['timestamp'])
        n_after = len(df)
        
        if n_before > n_after:
            print(f"Removed {n_before - n_after} rows with invalid timestamps")
        
        # Sort by timestamp to ensure correct order for inter-arrival calculation
        df = df.sort_values('timestamp')
        
        return df
    
    def calculate_inter_arrival_times(self, df, group_by=None):
        df = df.copy()
        if group_by:
            # Calculate inter-arrival times within groups
            result_dfs = []
            for group_name, group_df in df.groupby(group_by):
                sorted_group = group_df.sort_values('timestamp')
                # Calculate inter-arrival time in milliseconds
                sorted_group['inter_arrival_time'] = sorted_group['timestamp'].diff() * 1000
                result_dfs.append(sorted_group)
            df = pd.concat(result_dfs)
        else:
            # Calculate inter-arrival times for all packets
            df['inter_arrival_time'] = df['timestamp'].diff() * 1000
        
        # Remove rows with NaN inter-arrival times
        df = df.dropna(subset=['inter_arrival_time'])
        return df
    
    def analyze_ip_statistics(self, df):
        # Source IP statistics
        source_counts = df['Source'].value_counts()
        top_sources = source_counts.head(10)
        
        # Destination IP statistics
        dest_counts = df['Destination'].value_counts()
        top_dests = dest_counts.head(10)
        
        # Protocol statistics
        if 'Protocol' in df.columns:
            protocol_counts = df['Protocol'].value_counts()
            top_protocols = protocol_counts.head(10)
        else:
            protocol_counts = None
            top_protocols = None
        
        # Print summary
        print("\nIP Address Statistics:")
        print(f"Unique Source IPs: {len(source_counts)}")
        print(f"Unique Destination IPs: {len(dest_counts)}")
        
        print("\nTop Source IPs:")
        for ip, count in top_sources.items():
            print(f"{ip}: {count} packets ({count/len(df)*100:.2f}%)")
        
        print("\nTop Destination IPs:")
        for ip, count in top_dests.items():
            print(f"{ip}: {count} packets ({count/len(df)*100:.2f}%)")
        
        if protocol_counts is not None:
            print("\nTop Protocols:")
            for protocol, count in top_protocols.items():
                print(f"{protocol}: {count} packets ({count/len(df)*100:.2f}%)")
        
        return {
            'source_counts': source_counts,
            'dest_counts': dest_counts,
            'protocol_counts': protocol_counts
        }
    
    def extract_upload_download_traffic(self, df, client_ip):
        # Uplink: client is source
        uplink_df = df[df['Source'] == client_ip].copy()
        uplink_df['direction'] = 'uplink'
        # Downlink: client is destination
        downlink_df = df[df['Destination'] == client_ip].copy()
        downlink_df['direction'] = 'downlink'
        print(f"Uplink packets: {len(uplink_df)}")
        print(f"Downlink packets: {len(downlink_df)}")
        return uplink_df, downlink_df
    
    def split_data_by_protocol(self, df):
        if 'Protocol' not in df.columns:
            print("No 'Protocol' column in dataframe")
            return {}
        
        protocols = df['Protocol'].unique()
        result = {}
        
        for protocol in protocols:
            protocol_df = df[df['Protocol'] == protocol].copy()
            if len(protocol_df) > 0:
                result[protocol] = protocol_df
                print(f"Protocol {protocol}: {len(protocol_df)} packets")
        
        return result
    
    def prepare_data_for_modeling(self, df, max_value=None, min_value=None):
        if 'inter_arrival_time' not in df.columns:
            raise ValueError("No 'inter_arrival_time' column in dataframe")
        
        # Extract data
        data = df['inter_arrival_time'].values
        # Apply filters if provided
        if max_value is not None:
            data = data[data <= max_value]
        if min_value is not None:
            data = data[data >= min_value]
        print(f"Prepared data: {len(data)} values")
        print(f"Mean: {np.mean(data):.2f} ms, Median: {np.median(data):.2f} ms")
        print(f"Min: {np.min(data):.2f} ms, Max: {np.max(data):.2f} ms")
        return data

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Process 5G network packet data')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--app', type=str, choices=['Zepeto', 'Roblox', 'all'], default='all',
                      help='Metaverse application to analyze')
    parser.add_argument('--max-files', type=int, default=None, help='Maximum number of files to load')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for results')
    parser.add_argument('--client-ip', type=str, help='Client IP address for up/down traffic analysis')

    args = parser.parse_args()
    # Create output directory
    create_directory(args.output_dir)    
    # Initialize processor
    processor = PacketDataProcessor(args.data_dir)
    # Load data
    if args.app == 'all':
        df = processor.load_metaverse_data(max_files=args.max_files)
    else:
        df = processor.load_metaverse_data(app_name=args.app, max_files=args.max_files)
    if df is None or len(df) == 0:
        print("No data loaded. Exiting.")
        return
    # Process data
    df = processor.parse_timestamps(df)
    # Analyze IP statistics to help identify client IP if not provided
    ip_stats = processor.analyze_ip_statistics(df)
    # Calculate inter-arrival times
    df = processor.calculate_inter_arrival_times(df)
    # Save processed data
    output_csv = os.path.join(args.output_dir, f"processed_data_{args.app}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved processed data to {output_csv}")

if __name__ == "__main__":
    main()