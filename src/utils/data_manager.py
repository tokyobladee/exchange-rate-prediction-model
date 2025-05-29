#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

class DataManager:

    def __init__(self):
        self.base_dirs = ['data', 'models', 'plots', 'reports', 'backups']
        self.setup_directories()
        self.data_registry = {}
        self.load_registry()
    
    def setup_directories(self):
        
        for base_dir in self.base_dirs:
            os.makedirs(base_dir, exist_ok=True)
            
            os.makedirs(f'{base_dir}/current', exist_ok=True)
            os.makedirs(f'{base_dir}/archive', exist_ok=True)
            os.makedirs(f'{base_dir}/versions', exist_ok=True)
    
    def load_registry(self):
        
        registry_file = 'data/data_registry.json'
        
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                self.data_registry = json.load(f)
        else:
            self.data_registry = {}
    
    def save_registry(self):
        
        registry_file = 'data/data_registry.json'
        
        with open(registry_file, 'w') as f:
            json.dump(self.data_registry, f, indent=2, default=str)
    
    def create_backup(self, pair: str, backup_type: str = 'auto'):
        
        clean_pair = pair.replace('=X', '').replace('/', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f" CREATING BACKUP FOR {pair}")
        print("=" * 50)
        
        backup_dir = f'backups/{clean_pair}/{timestamp}_{backup_type}'
        os.makedirs(backup_dir, exist_ok=True)
        
        data_files = [
            f'data/{clean_pair}_processed.csv',
            f'data/{clean_pair}_raw.csv',
            f'data/{clean_pair}_predictions.csv'
        ]
        
        backed_up_files = []
        
        for file_path in data_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                backup_path = f'{backup_dir}/{filename}'
                shutil.copy2(file_path, backup_path)
                backed_up_files.append(filename)
        
        model_files = [
            f'models/{clean_pair}_traditional.joblib',
            f'models/{clean_pair}_neural.pth',
            f'models/{clean_pair}_pytorch_models.pth'
        ]
        
        for file_path in model_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                backup_path = f'{backup_dir}/{filename}'
                shutil.copy2(file_path, backup_path)
                backed_up_files.append(filename)
        
        report_files = [
            f'reports/{clean_pair}_performance.json',
            f'reports/{clean_pair}_analysis.txt'
        ]
        
        for file_path in report_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                backup_path = f'{backup_dir}/{filename}'
                shutil.copy2(file_path, backup_path)
                backed_up_files.append(filename)
        
        if pair not in self.data_registry:
            self.data_registry[pair] = {'backups': []}
        
        backup_info = {
            'timestamp': timestamp,
            'backup_type': backup_type,
            'backup_dir': backup_dir,
            'files_backed_up': backed_up_files,
            'created_at': datetime.now().isoformat()
        }
        
        self.data_registry[pair]['backups'].append(backup_info)
        self.save_registry()
        
        print(f" Backup created: {backup_dir}")
        print(f" Files backed up: {len(backed_up_files)}")
        
        return backup_dir
    
    def restore_backup(self, pair: str, backup_timestamp: str = None):
        
        clean_pair = pair.replace('=X', '').replace('/', '')
        
        print(f" RESTORING BACKUP FOR {pair}")
        print("=" * 50)
        
        if pair not in self.data_registry or not self.data_registry[pair]['backups']:
            print(f" No backups found for {pair}")
            return False
        
        backups = self.data_registry[pair]['backups']
        
        if backup_timestamp:
            backup_info = next((b for b in backups if b['timestamp'] == backup_timestamp), None)
            if not backup_info:
                print(f" Backup {backup_timestamp} not found")
                return False
        else:
            backup_info = max(backups, key=lambda x: x['timestamp'])
        
        backup_dir = backup_info['backup_dir']
        
        if not os.path.exists(backup_dir):
            print(f" Backup directory not found: {backup_dir}")
            return False
        
        restored_files = []
        
        for filename in backup_info['files_backed_up']:
            backup_file = f'{backup_dir}/{filename}'
            
            if filename.endswith('.csv'):
                restore_path = f'data/{filename}'
            elif filename.endswith(('.joblib', '.pth')):
                restore_path = f'models/{filename}'
            elif filename.endswith('.json') or filename.endswith('.txt'):
                restore_path = f'reports/{filename}'
            else:
                continue
            
            if os.path.exists(backup_file):
                if os.path.exists(restore_path):
                    current_backup = f'{restore_path}.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                    shutil.copy2(restore_path, current_backup)
                
                shutil.copy2(backup_file, restore_path)
                restored_files.append(filename)
        
        print(f" Restored {len(restored_files)} files from backup {backup_info['timestamp']}")
        print(f" Backup date: {backup_info['created_at'][:10]}")
        
        return True
    
    def list_backups(self, pair: str = None):
        
        print(" AVAILABLE BACKUPS")
        print("=" * 50)
        
        if pair:
            pairs_to_check = [pair] if pair in self.data_registry else []
        else:
            pairs_to_check = list(self.data_registry.keys())
        
        if not pairs_to_check:
            print("No backups found")
            return
        
        for pair_name in pairs_to_check:
            if 'backups' not in self.data_registry[pair_name]:
                continue
                
            backups = self.data_registry[pair_name]['backups']
            
            print(f"\n {pair_name}:")
            print("-" * 30)
            
            for i, backup in enumerate(sorted(backups, key=lambda x: x['timestamp'], reverse=True), 1):
                print(f"{i:2d}. {backup['timestamp']} ({backup['backup_type']})")
                print(f"      {len(backup['files_backed_up'])} files")
                print(f"      {backup['created_at'][:10]}")
    
    def preserve_data_before_analysis(self, pair: str, analysis_type: str = 'training'):
        
        clean_pair = pair.replace('=X', '').replace('/', '')
        
        data_files = [
            f'data/{clean_pair}_processed.csv',
            f'models/{clean_pair}_traditional.joblib',
            f'models/{clean_pair}_neural.pth'
        ]
        
        has_existing_data = any(os.path.exists(f) for f in data_files)
        
        if has_existing_data:
            print(f" Existing data found for {pair}")
            
            preserve = True
            
            if preserve:
                backup_dir = self.create_backup(pair, f'{analysis_type}_auto')
                print(f" Data preserved before {analysis_type}")
                return backup_dir
            else:
                print(f"  Proceeding without backup (data may be overwritten)")
                return None
        else:
            print(f" No existing data found for {pair} - proceeding with fresh analysis")
            return None
    
    def get_data_history(self, pair: str):
        
        clean_pair = pair.replace('=X', '').replace('/', '')
        
        print(f" DATA HISTORY FOR {pair}")
        print("=" * 50)
        
        if pair not in self.data_registry:
            print("No data history found")
            return
        
        registry_info = self.data_registry[pair]
        
        current_files = []
        data_files = [
            f'data/{clean_pair}_processed.csv',
            f'models/{clean_pair}_traditional.joblib',
            f'models/{clean_pair}_neural.pth',
            f'reports/{clean_pair}_performance.json'
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                current_files.append({
                    'file': os.path.basename(file_path),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        print(" CURRENT FILES:")
        for file_info in current_files:
            size_mb = file_info['size'] / (1024 * 1024)
            print(f"   {file_info['file']:30s} | {size_mb:6.2f} MB | {file_info['modified'][:10]}")
        
        if 'backups' in registry_info:
            print(f"\n BACKUP HISTORY ({len(registry_info['backups'])} backups):")
            for backup in sorted(registry_info['backups'], key=lambda x: x['timestamp'], reverse=True):
                print(f"   {backup['timestamp']} | {backup['backup_type']:10s} | {len(backup['files_backed_up'])} files")
        
        return registry_info
    
    def cleanup_old_backups(self, pair: str = None, keep_days: int = 30):
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        print(f"CLEANING UP BACKUPS OLDER THAN {keep_days} DAYS")
        print("=" * 60)
        
        pairs_to_clean = [pair] if pair else list(self.data_registry.keys())
        
        total_cleaned = 0
        
        for pair_name in pairs_to_clean:
            if pair_name not in self.data_registry or 'backups' not in self.data_registry[pair_name]:
                continue
            
            backups = self.data_registry[pair_name]['backups']
            backups_to_remove = []
            
            for backup in backups:
                backup_date = datetime.fromisoformat(backup['created_at'])
                
                if backup_date < cutoff_date:
                    backup_dir = backup['backup_dir']
                    if os.path.exists(backup_dir):
                        shutil.rmtree(backup_dir)
                        print(f"  Removed: {backup['timestamp']} ({pair_name})")
                        total_cleaned += 1
                    
                    backups_to_remove.append(backup)
            
            for backup in backups_to_remove:
                self.data_registry[pair_name]['backups'].remove(backup)
        
        self.save_registry()
        
        print(f" Cleaned up {total_cleaned} old backups")
    
    def export_data(self, pair: str, export_format: str = 'csv'):
        
        clean_pair = pair.replace('=X', '').replace('/', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f" EXPORTING DATA FOR {pair}")
        print("=" * 50)
        
        export_dir = f'exports/{clean_pair}_{timestamp}'
        os.makedirs(export_dir, exist_ok=True)
        
        exported_files = []
        
        data_file = f'data/{clean_pair}_processed.csv'
        if os.path.exists(data_file):
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
            
            if export_format == 'csv':
                export_path = f'{export_dir}/{clean_pair}_data.csv'
                data.to_csv(export_path)
            elif export_format == 'json':
                export_path = f'{export_dir}/{clean_pair}_data.json'
                data.to_json(export_path, orient='index', date_format='iso')
            elif export_format == 'excel':
                export_path = f'{export_dir}/{clean_pair}_data.xlsx'
                data.to_excel(export_path)
            
            exported_files.append(export_path)
        
        performance_file = f'reports/{clean_pair}_performance.json'
        if os.path.exists(performance_file):
            export_path = f'{export_dir}/{clean_pair}_performance.json'
            shutil.copy2(performance_file, export_path)
            exported_files.append(export_path)
        
        print(f" Exported {len(exported_files)} files to {export_dir}")
        
        return export_dir

def main():
    
    manager = DataManager()
    
    print(" TESTING DATA MANAGER")
    print("=" * 50)
    
    manager.create_backup('EURUSD=X', 'manual')
    
    manager.list_backups()
    
    manager.get_data_history('EURUSD=X')
    
    print("\n Data manager test completed!")

if __name__ == "__main__":
    main() 