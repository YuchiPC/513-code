# data_processor.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob
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
        df = df.copy()
        df['timestamp'] = df['Time'].apply(parse_timestamp)
        n_before = len(df)
        df = df.dropna(subset=['timestamp'])
        n_after = len(df)
        if n_before > n_after:
            print(f"Removed {n_before - n_after} rows with invalid timestamps")
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
                sorted_group['inter_arrival_time'] = sorted_group['timestamp'].diff()
                result_dfs.append(sorted_group)
            df = pd.concat(result_dfs)
        else:
            # Calculate inter-arrival times for all packets
            df['inter_arrival_time'] = df['timestamp'].diff()
        df = df.dropna(subset=['inter_arrival_time'])
        df['inter_arrival_ms'] = df['inter_arrival_time'].dt.total_seconds() * 1000
        return df
    
    def analyze_ip_statistics(self, df):
        # Source IP
        source_counts = df['Source'].value_counts()
        top_sources = source_counts.head(10)
        # Destination IP
        dest_counts = df['Destination'].value_counts()
        top_dests = dest_counts.head(10)
        # Protocol
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
        # Likely client IP
        likely_client_ip = None
        max_count = 0
        for ip in set(source_counts.index) | set(dest_counts.index):
            total_count = source_counts.get(ip, 0) + dest_counts.get(ip, 0)
            if total_count > max_count:
                max_count = total_count
                likely_client_ip = ip
        print(f"\nLikely client IP: {likely_client_ip} (appears in {max_count} packets)")
        return {
            'source_counts': source_counts,
            'dest_counts': dest_counts,
            'protocol_counts': protocol_counts,
            'likely_client_ip': likely_client_ip
        }
    
    def extract_upload_download_traffic(self, df, client_ip=None):
        if client_ip is None:
            ip_stats = self.analyze_ip_statistics(df)
            client_ip = ip_stats['likely_client_ip']
            print(f"Auto-detected client IP: {client_ip}")
        # Uplink
        uplink_df = df[df['Source'] == client_ip].copy()
        uplink_df['direction'] = 'uplink'
        # Downlink
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
        if 'inter_arrival_ms' not in df.columns:
            raise ValueError("No 'inter_arrival_ms' column in dataframe")
        data = df['inter_arrival_ms'].values
        if max_value is not None:
            data = data[data <= max_value]
        if min_value is not None:
            data = data[data >= min_value]
        print(f"Prepared data: {len(data)} values")
        print(f"Mean: {np.mean(data):.2f} ms, Median: {np.median(data):.2f} ms")
        print(f"Min: {np.min(data):.2f} ms, Max: {np.max(data):.2f} ms")
        return data
    
    def process_direction_specific_data(self, direction_df, direction, output_dir):
        if len(direction_df) == 0:
            print(f"No {direction} packets found")
            return None
        direction_df = self.calculate_inter_arrival_times(direction_df)
        protocol_dfs = self.split_data_by_protocol(direction_df)
        for protocol, protocol_df in protocol_dfs.items():
            if len(protocol_df) > 100:
                protocol_output = os.path.join(output_dir, f"processed_{direction}_{protocol}.csv")
                protocol_df.to_csv(protocol_output, index=False)
                print(f"Saved {direction} {protocol} data to {protocol_output}")
                data = self.prepare_data_for_modeling(protocol_df)
                fig_path = os.path.join(output_dir, f"{direction}_{protocol}_histogram.png")
                plt.figure(figsize=(10, 6))
                plt.hist(data, bins=50, alpha=0.7, density=True)
                plt.title(f"{direction.capitalize()} {protocol} Inter-arrival Time Distribution")
                plt.xlabel("Inter-arrival Time (ms)")
                plt.ylabel("Density")
                plt.grid(True, alpha=0.3)
                plt.savefig(fig_path)
                plt.close()
                print(f"Saved histogram to {fig_path}")

        output_csv = os.path.join(output_dir, f"processed_{direction}_all.csv")
        direction_df.to_csv(output_csv, index=False)
        print(f"Saved {direction} data to {output_csv}")
        return direction_df

def main():
    data_dir = 'data/5G_Traffic_Datasets'
    app_name = 'Zepeto'
    output_dir = 'data/processed'
    create_directory(output_dir)
    
    processor = PacketDataProcessor(data_dir)
    df = processor.load_metaverse_data(app_name=app_name)
    if df is None or len(df) == 0:
        print("No data loaded. Exiting.")
        return
    df = processor.parse_timestamps(df)
    uplink_df, downlink_df = processor.extract_upload_download_traffic(df)
    processor.process_direction_specific_data(uplink_df, 'uplink', output_dir)
    processor.process_direction_specific_data(downlink_df, 'downlink', output_dir)
    print("Processing complete!")

if __name__ == "__main__":
    main()