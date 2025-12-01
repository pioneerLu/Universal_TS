import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    def __init__(self, dataset_path, flag='train', split=0.8, 
                 input_len=None, output_len=None, norm=True, stride=1):
        self.data_dir = dataset_path
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len + output_len
        
        assert flag in ['train', 'val']
        assert split >= 0 and split <= 1.0
        
        self.type_map = {'train': 0, 'val': 1}
        self.set_type = self.type_map[flag]
        self.flag = flag
        self.split = split
        self.norm = norm
        self.stride = stride
        
        self.data_list = []       
        self.n_window_list = []  
        self.scalers = []          
        
        self.__read_data__()
        
    def __read_data__(self):

        print(f'Loading NPY files from {self.data_dir}...')
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        
        for file_name in tqdm(npy_files):
            file_path = os.path.join(self.data_dir, file_name)
            #(N, T, 1)
            raw_data = np.load(file_path, allow_pickle=True)
            
            for i in range(raw_data.shape[0]):
                ts_data = raw_data[i]  #  (T, 1)
                num_train = int(len(ts_data) * self.split)
                
                # Define borders for train/val sets
                border1s = [0, num_train - self.seq_len]
                border2s = [num_train, len(ts_data)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]
                
                if border1 >= border2 - self.seq_len:
                    continue  
                
                if self.norm:
                    scaler = StandardScaler()
                    # Fit on training portion only
                    train_data = ts_data[border1s[0]:border2s[0]]
                    scaler.fit(train_data)
                    ts_data = scaler.transform(ts_data)
                    self.scalers.append(scaler)
                
                # relevant portion based on train/val flag
                data = ts_data[border1:border2]
                
                #  number of windows for this time series
                n_window = (len(data) - self.seq_len) // self.stride + 1
                
                if n_window < 1:
                    continue  #  if can't create at least one window
                
                # Store processed data and update window count
                self.data_list.append(data)
                prev_windows = self.n_window_list[-1] if self.n_window_list else 0
                self.n_window_list.append(prev_windows + n_window)
                
        print(f'Processed {len(self.data_list)} time series with {self.n_window_list[-1]} total windows')
    
    def __getitem__(self, index):
        # Find which time series this index belongs to
        dataset_index = 0
        while dataset_index < len(self.n_window_list) and index >= self.n_window_list[dataset_index]:
            dataset_index += 1
        
        # Adjust index to be relative to the selected time series
        if dataset_index > 0:
            index = index - self.n_window_list[dataset_index - 1]
        
        # Calculate start positions for this window
        data = self.data_list[dataset_index]
        n_timepoint = (len(data) - self.seq_len) // self.stride + 1
        s_begin = index % n_timepoint
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        
        # Split the sequence into input and output portions
        seq_x = data[s_begin:s_begin+self.input_len]
        seq_y = data[s_begin+self.input_len:s_end]
        
        return dict(seq_x=seq_x, seq_y=seq_y)
    
    def __len__(self):
        if not self.n_window_list:
            return 0
        return self.n_window_list[-1]
    
    def inverse_transform(self, data, dataset_index):
        if self.norm and dataset_index < len(self.scalers):
            return self.scalers[dataset_index].inverse_transform(data)
        return data
    


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class SingleVarDataset(Dataset):
    def __init__(self, data_path, flag='train', size=None, 
                 normalize=True, target='OT', split=[0.5, 0.5, 0.0],
                 stride=1,zero_shot=0):
        assert flag in ['train', 'val', 'test']
        assert target in ['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        
        self.data_path = data_path
        self.flag = flag
        self.target = target
        self.normalize = normalize
        self.stride = stride
        self.split = split
        self.zero_shot = zero_shot
        # Size 
        if size is None:
            self.input_len = 96     
            self.output_len = 1    
        else:
            self.input_len, self.output_len = size
        self.seq_len = self.input_len + self.output_len
        
        # Type 
        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = self.type_map[flag]
        
        self._read_data()
    
    def _read_data(self):
        """Read and preprocess the ETT data"""
        # Read data
        df_raw = pd.read_csv(self.data_path)
        
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # Extract target variable only (single variable forecasting)
        data = df_raw[[self.target]].values
        
        num_samples = len(data)
        train_end = int(num_samples * self.split[0])
        val_end = int(num_samples * (self.split[0]+self.split[1]))
        
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, num_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Normalization
        self.scaler = StandardScaler()
        if self.normalize:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data) ##only training!
            data = self.scaler.transform(data)
            
        self.data = data[border1:border2]

        # if self.zero_shot ==1:
        #     self.data = data[0:border2]
        # else:
        #     self.data = data[border1:border2]
        
        # Calculate number of available samples
        self.n_samples = (len(self.data) - self.seq_len) // self.stride + 1
        
        print(f"[{self.flag} set] Total available samples: {self.n_samples}")
    
    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        
        if s_end > len(self.data):
            s_begin = len(self.data) - self.seq_len
            s_end = len(self.data)
        
        # Extract sequence
        seq = self.data[s_begin:s_end]
        
        seq_x = seq[:self.input_len]
        seq_y = seq[self.input_len:self.seq_len]
        
        return dict(seq_x=seq_x, seq_y=seq_y)
    
    def __len__(self):
        return self.n_samples
    
    def inverse_transform(self, data):
        if self.normalize:
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            return self.scaler.inverse_transform(data)
        return data




class MultiVariateTimeSeriesDataset(Dataset):
    def __init__(self, data_path, flag='train', size=None, 
                 features='all', target=None, normalize=True,
                 split=[0.7, 0.2, 0.1], stride=1):

        assert flag in ['train', 'val', 'test']
        
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.normalize = normalize
        self.stride = stride
        
        if size is None:
            self.input_len = 100
            self.output_len = 1   
        else:
            self.input_len, self.output_len = size
        self.seq_len = self.input_len + self.output_len
        
        # Type parameters
        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = self.type_map[flag]
        self.split = split
        
        self._read_data()

    ##指定column名称的做法
    # def _read_data(self):
    #     df_raw = pd.read_csv(self.data_path, header=0)
        
    #     # Select features
    #     if self.features == 'all':
    #         self.feature_names = df_raw.columns.tolist()
    #     else:
    #         self.feature_names = self.features if isinstance(self.features, list) else [self.features]
    #         assert all(feat in df_raw.columns for feat in self.feature_names), "Some features not found in data"
        
    #     if self.target is None:
    #         self.target_names = self.feature_names
    #     else:
    #         self.target_names = self.target if isinstance(self.target, list) else [self.target]
    #         assert all(t in df_raw.columns for t in self.target_names), "Some target variables not found in data"
        
    #     df_data = df_raw[self.feature_names]
        
    #     data = df_data.values
        
    #     # split borders
    #     num_samples = len(data)
    #     train_end = int(num_samples * self.split[0])
    #     val_end = int(num_samples * (self.split[0] + self.split[1]))
        
    #     # Define borders
    #     border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
    #     border2s = [train_end, val_end, num_samples]
    #     border1 = border1s[self.set_type]
    #     border2 = border2s[self.set_type]
        
    #     # Normalization
    #     self.scalers = {}
    #     if self.normalize:
    #         train_data = data[border1s[0]:border2s[0]]
    #         # scaler for each feature
    #         for i, feature_name in enumerate(self.feature_names):
    #             self.scalers[feature_name] = StandardScaler()
    #             self.scalers[feature_name].fit(train_data[:, i:i+1])
    #             data[:, i:i+1] = self.scalers[feature_name].transform(data[:, i:i+1])
        
    #     self.data = data[border1:border2]
        
    #     # create index mapping from feature names to column indices
    #     self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        
    #     # map target names to indices
    #     self.target_indices = [self.feature_indices[name] for name in self.target_names]
        
    #     self.n_samples = (len(self.data) - self.seq_len) // self.stride + 1
        
    #     print(f"[{self.flag} set] Total available samples: {self.n_samples}")
    #     print(f"Features: {self.feature_names}")
    #     print(f"Target variables: {self.target_names}")
    
    #指定索引的做法
    def _read_data(self):

        df_raw = pd.read_csv(self.data_path, header=None)
        
        # 基于索引选择特征
        if self.features == 'all':
            self.feature_indices = list(range(df_raw.shape[1]))
        else:
            self.feature_indices = self.features if isinstance(self.features, list) else [self.features]
            assert all(0 <= idx < df_raw.shape[1] for idx in self.feature_indices), "Some feature indices are out of range"
        
        if self.target is None:
            self.target_indices = self.feature_indices
        else:
            self.target_indices = self.target if isinstance(self.target, list) else [self.target]
            assert all(0 <= idx < df_raw.shape[1] for idx in self.target_indices), "Some target indices are out of range"
        
        # 基于索引选择特征列
        df_data = df_raw.iloc[:, self.feature_indices]
        
        data = df_data.values
        
        # split borders
        num_samples = len(data)
        train_end = int(num_samples * self.split[0])
        val_end = int(num_samples * (self.split[0] + self.split[1]))
        
        # Define borders
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, num_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Normalization
        self.scalers = {}
        if self.normalize:
            train_data = data[border1s[0]:border2s[0]]
            # scaler for each feature
            for i, feature_idx in enumerate(range(len(self.feature_indices))):
                self.scalers[feature_idx] = StandardScaler()
                self.scalers[feature_idx].fit(train_data[:, i:i+1])
                data[:, i:i+1] = self.scalers[feature_idx].transform(data[:, i:i+1])
        
        self.data = data[border1:border2]
        
        # 将target_indices映射到feature_indices中的位置
        self.target_positions = [self.feature_indices.index(idx) if idx in self.feature_indices 
                                else None for idx in self.target_indices]
        assert None not in self.target_positions, "Some target indices are not in selected features"
        
        self.n_samples = (len(self.data) - self.seq_len) // self.stride + 1
        
        print(f"[{self.flag} set] Total available samples: {self.n_samples}")
        print(f"Features indices: {self.feature_indices}")
        print(f"Target indices: {self.target_indices}")

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        
        # Ensure the index is valid
        if s_end > len(self.data):
            s_begin = len(self.data) - self.seq_len
            s_end = len(self.data)
        
        # Extract sequence
        seq = self.data[s_begin:s_end]
        
        # Split into x and y
        seq_x = seq[:self.input_len]
        seq_y = seq[self.input_len:, self.target_positions]  
        
        return dict(seq_x=seq_x, seq_y=seq_y)
    
    def __len__(self):
        return self.n_samples
    
    def inverse_transform(self, data, variable_idx=None):
        if not self.normalize:
            return data
            
        if variable_idx is None:

            variable_position = self.target_positions[0]
        else:
            # 索引在feature_indices中的位置
            variable_position = self.feature_indices.index(variable_idx)
                
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
            return self.scalers[variable_position].inverse_transform(data)
    def get_feature_indices(self):
        return self.feature_indices

    def get_target_indices(self):
        return self.target_indices



class MultiVariateNPYTimeSeriesDataset(Dataset):
    def __init__(self, data_path, flag='train', size=None, 
                 features='all', target=None, normalize=True,
                 split=[0.7, 0.2, 0.1], stride=1):
        assert flag in ['train', 'val', 'test']
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.normalize = normalize
        self.stride = stride
        if size is None:
            self.input_len = 100
            self.output_len = 1   
        else:
            self.input_len, self.output_len = size
        self.seq_len = self.input_len + self.output_len
        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = self.type_map[flag]
        self.split = split
        self._read_data()

    def _read_data(self):
        arr = np.load(self.data_path)
        assert arr.ndim == 2, "npy数据必须为二维 (T, C)"
        T, C = arr.shape
        if self.features == 'all':
            self.feature_indices = list(range(C))
        else:
            self.feature_indices = self.features if isinstance(self.features, list) else [self.features]
            assert all(0 <= idx < C for idx in self.feature_indices), "Some feature indices are out of range"
        if self.target is None:
            self.target_indices = self.feature_indices
        else:
            self.target_indices = self.target if isinstance(self.target, list) else [self.target]
            assert all(0 <= idx < C for idx in self.target_indices), "Some target indices are out of range"
        data = arr[:, self.feature_indices]
        num_samples = len(data)
        train_end = int(num_samples * self.split[0])
        val_end = int(num_samples * (self.split[0] + self.split[1]))
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, num_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.scalers = {}
        if self.normalize:
            train_data = data[border1s[0]:border2s[0]]
            for i, feature_idx in enumerate(range(len(self.feature_indices))):
                self.scalers[feature_idx] = StandardScaler()
                self.scalers[feature_idx].fit(train_data[:, i:i+1])
                data[:, i:i+1] = self.scalers[feature_idx].transform(data[:, i:i+1])
        self.data = data[border1:border2]
        self.target_positions = [self.feature_indices.index(idx) if idx in self.feature_indices 
                                else None for idx in self.target_indices]
        assert None not in self.target_positions, "Some target indices are not in selected features"
        self.n_samples = (len(self.data) - self.seq_len) // self.stride + 1
        print(f"[{self.flag} set] Total available samples: {self.n_samples}")
        print(f"Features indices: {self.feature_indices}")
        print(f"Target indices: {self.target_indices}")

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        if s_end > len(self.data):
            s_begin = len(self.data) - self.seq_len
            s_end = len(self.data)
        seq = self.data[s_begin:s_end]
        seq_x = seq[:self.input_len]
        seq_y = seq[self.input_len:, self.target_positions]  
        return dict(seq_x=seq_x, seq_y=seq_y)
    
    def __len__(self):
        return self.n_samples
    
    def inverse_transform(self, data, variable_idx=None):
        if not self.normalize:
            return data
        if variable_idx is None:
            variable_position = self.target_positions[0]
        else:
            variable_position = self.feature_indices.index(variable_idx)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return self.scalers[variable_position].inverse_transform(data)
    
    def get_feature_indices(self):
        return self.feature_indices

    def get_target_indices(self):
        return self.target_indices


