import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    def __init__(self, dataset_path, flag='train', split=0.8, 
                 input_len=None, output_len=None, norm=True, stride=1, cache_dir=None):
        self.data_dir = dataset_path
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len + output_len
        self.cache_dir = cache_dir
        
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
        self.use_cache = False
        self._array_cache = {}

        cache_info = cache_dir and os.path.exists(os.path.join(cache_dir, "cache_info.json"))
        chunks_info = cache_dir and os.path.exists(os.path.join(cache_dir, "chunks_info.json"))
        if cache_info or chunks_info:
            self._load_from_cache()
        else:
            self.__read_data__()
        
    def __read_data__(self):
        print(f'Loading NPY files metadata (lazy loading)...')
        if os.path.isfile(self.data_dir) and self.data_dir.endswith('.npy'):
            npy_files = [os.path.basename(self.data_dir)]
            self.data_dir = os.path.dirname(os.path.abspath(self.data_dir)) or '.'
        elif os.path.isdir(self.data_dir):
            npy_files = sorted(f for f in os.listdir(self.data_dir) if f.endswith('.npy'))
        else:
            raise FileNotFoundError(f"data path is neither a .npy file nor a directory: {self.data_dir}")
        
        for file_name in tqdm(npy_files, desc="Processing files"):
            file_path = os.path.join(self.data_dir, file_name)
            
            try:
                file_size = os.path.getsize(file_path)
                use_mmap = file_size > 500 * 1024 * 1024
                raw_data = np.load(file_path, mmap_mode='r' if use_mmap else None, allow_pickle=True)
                self._array_cache[file_path] = raw_data
            except Exception as e:
                print(f'Warning: Failed to load {file_name}: {e}')
                raw_data = np.load(file_path, allow_pickle=True)
                use_mmap = False
                self._array_cache[file_path] = raw_data
            
            for i in range(raw_data.shape[0]):
                ts_data = np.array(raw_data[i]).reshape(-1, 1)
                num_train = int(len(ts_data) * self.split)
                
                border1s = [0, num_train - self.seq_len]
                border2s = [num_train, len(ts_data)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]
                
                if border1 >= border2 - self.seq_len:
                    del ts_data
                    continue  
                
                scaler_params = None
                if self.norm:
                    scaler = StandardScaler()
                    train_data = ts_data[border1s[0]:border2s[0]]
                    scaler.fit(train_data)
                    scaler_params = {
                        'mean': scaler.mean_.copy(),
                        'scale': scaler.scale_.copy()
                    }
                    del train_data, scaler
                
                data_len = border2 - border1
                n_window = (data_len - self.seq_len) // self.stride + 1
                
                if n_window < 1:
                    del ts_data
                    continue
                
                self.data_list.append({
                    'file_path': file_path,
                    'item_idx': i,
                    'border1': border1,
                    'border2': border2,
                    'data_len': len(ts_data),
                    'scaler_params': scaler_params,
                    'use_mmap': use_mmap
                })
                
                prev_windows = self.n_window_list[-1] if self.n_window_list else 0
                self.n_window_list.append(prev_windows + n_window)
                
                del ts_data
            
            del raw_data
                
        print(f'Processed {len(self.data_list)} time series with {self.n_window_list[-1] if self.n_window_list else 0} total windows (lazy loading enabled)')
    
    def _load_npy(self, file_path, use_mmap):
        arr = self._array_cache.get(file_path)
        if arr is None:
            arr = np.load(file_path, mmap_mode="r" if use_mmap else None, allow_pickle=True)
            self._array_cache[file_path] = arr
        return arr

    def _load_from_cache(self):
        import json

        cache_info_path = os.path.join(self.cache_dir, "cache_info.json")
        chunks_info_path = os.path.join(self.cache_dir, "chunks_info.json")
        if os.path.exists(cache_info_path):
            with open(cache_info_path, "r") as f:
                self.cache_info = json.load(f)
        elif os.path.exists(chunks_info_path):
            with open(chunks_info_path, "r") as f:
                chunks_data = json.load(f)
            self.cache_info = {
                "input_len": chunks_data.get("input_len", self.input_len),
                "output_len": chunks_data.get("output_len", self.output_len),
                "split": chunks_data.get("split", self.split),
                "norm": chunks_data.get("norm", self.norm),
                "stride": chunks_data.get("stride", self.stride),
                "num_train": chunks_data.get("total_train", 0),
                "num_val": chunks_data.get("total_val", 0),
            }
        else:
            raise FileNotFoundError(f"no cache_info.json or chunks_info.json in {self.cache_dir}")
        
        if self.cache_info['input_len'] != self.input_len:
            raise ValueError(f"Cache input_len ({self.cache_info['input_len']}) != current input_len ({self.input_len})")
        if self.cache_info['output_len'] != self.output_len:
            raise ValueError(f"Cache output_len ({self.cache_info['output_len']}) != current output_len ({self.output_len})")
        if abs(self.cache_info['split'] - self.split) > 1e-6:
            raise ValueError(f"Cache split ({self.cache_info['split']}) != current split ({self.split})")
        if self.cache_info['norm'] != self.norm:
            raise ValueError(f"Cache norm ({self.cache_info['norm']}) != current norm ({self.norm})")
        if self.cache_info['stride'] != self.stride:
            raise ValueError(f"Cache stride ({self.cache_info['stride']}) != current stride ({self.stride})")
        
        chunks_info_path = os.path.join(self.cache_dir, 'chunks_info.json')
        if os.path.exists(chunks_info_path):
            with open(chunks_info_path, 'r') as f:
                chunks_data = json.load(f)
            
            chunks = [c for c in chunks_data['chunks'] if c['type'] == self.flag]
            chunks.sort(key=lambda x: x['chunk_idx'])
            
            if not chunks:
                raise FileNotFoundError(f"No {self.flag} chunks found in chunks_info.json")
            
            self.chunk_data_list = []
            self.chunk_offsets = [0]
            total_windows = 0
            
            for chunk in chunks:
                chunk_path = os.path.join(self.cache_dir, chunk['file'])
                if not os.path.exists(chunk_path):
                    raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
                
                chunk_data = np.load(chunk_path, mmap_mode='r')
                self.chunk_data_list.append(chunk_data)
                total_windows += chunk['num_windows']
                self.chunk_offsets.append(total_windows)
            
            self.use_cache = True
            self.use_chunks = True
            print(f"Loaded {len(chunks)} {self.flag} chunks with {total_windows:,} total windows (using mmap)")
        else:
            cache_file = 'train_windows.npy' if self.flag == 'train' else 'val_windows.npy'
            cache_path = os.path.join(self.cache_dir, cache_file)
            
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Cache file not found: {cache_path}")
            
            self.cache_data = np.load(cache_path, mmap_mode='r')
            self.use_cache = True
            self.use_chunks = False
            
            num_windows = self.cache_info['num_train'] if self.flag == 'train' else self.cache_info['num_val']
            print(f"Loaded {num_windows:,} {self.flag} windows from cache (using mmap)")
    
    def __getitem__(self, index):
        if self.use_cache:
            if self.use_chunks:
                chunk_idx = 0
                for i in range(len(self.chunk_offsets) - 1):
                    if index < self.chunk_offsets[i + 1]:
                        chunk_idx = i
                        break
                local_index = index - self.chunk_offsets[chunk_idx]
                window = np.asarray(self.chunk_data_list[chunk_idx][local_index], dtype=np.float32)
            else:
                window = np.asarray(self.cache_data[index], dtype=np.float32)
            seq_x = window[:self.input_len]
            seq_y = window[self.input_len:]
            return dict(seq_x=seq_x, seq_y=seq_y)

        dataset_index = 0
        while dataset_index < len(self.n_window_list) and index >= self.n_window_list[dataset_index]:
            dataset_index += 1
        if dataset_index > 0:
            index = index - self.n_window_list[dataset_index - 1]

        meta = self.data_list[dataset_index]
        raw_data = self._load_npy(meta["file_path"], meta["use_mmap"])
        ts_data = np.asarray(raw_data[meta["item_idx"]], dtype=np.float32).reshape(-1, 1)

        if self.norm and meta["scaler_params"] is not None:
            params = meta["scaler_params"]
            ts_data = (ts_data - params["mean"].astype(np.float32)) / params["scale"].astype(np.float32)

        data = ts_data[meta["border1"]:meta["border2"]]
        n_timepoint = (len(data) - self.seq_len) // self.stride + 1
        s_begin = self.stride * (index % max(n_timepoint, 1))
        s_end = s_begin + self.seq_len
        seq_x = np.asarray(data[s_begin:s_begin + self.input_len], dtype=np.float32)
        seq_y = np.asarray(data[s_begin + self.input_len:s_end], dtype=np.float32)
        return dict(seq_x=seq_x, seq_y=seq_y)
    
    def __len__(self):
        if self.use_cache:
            if self.use_chunks:
                return self.chunk_offsets[-1]
            else:
                return len(self.cache_data)
        else:
            if not self.n_window_list:
                return 0
            return self.n_window_list[-1]
    
    def inverse_transform(self, data, dataset_index):
        if self.norm and dataset_index < len(self.data_list):
            meta = self.data_list[dataset_index]
            if meta['scaler_params'] is not None:
                params = meta['scaler_params']
                return data * params['scale'] + params['mean']
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
        df_raw = pd.read_csv(self.data_path)
        if "date" in df_raw.columns:
            df_raw["date"] = pd.to_datetime(df_raw["date"])
        if self.target not in df_raw.columns:
            numeric = df_raw.select_dtypes(include=["number"])
            if numeric.shape[1] == 0:
                raise ValueError(f"no numeric column in {self.data_path}")
            self.target = numeric.columns[-1]
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
        
        seq_x = np.asarray(seq[:self.input_len], dtype=np.float32)
        seq_y = np.asarray(seq[self.input_len:self.seq_len], dtype=np.float32)
        
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
        peek = pd.read_csv(self.data_path, header=None, nrows=1)

        def _numeric_row(row):
            for v in row.tolist():
                try:
                    float(v)
                except (TypeError, ValueError):
                    return False
            return True

        df_raw = pd.read_csv(self.data_path, header=None if _numeric_row(peek.iloc[0]) else 0)
        if df_raw.columns.dtype == object:
            drop_cols = [c for c in df_raw.columns if str(c).lower() in ("date", "time", "timestamp")]
            if drop_cols:
                df_raw = df_raw.drop(columns=drop_cols)
            df_raw = df_raw.apply(pd.to_numeric, errors="coerce")
        
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
        seq_x = np.asarray(seq[:self.input_len], dtype=np.float32)
        seq_y = np.asarray(seq[self.input_len:, self.target_positions], dtype=np.float32)
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
        
        scaler = self.scalers.get(variable_position)
        if scaler is None:
            return data
        arr = np.asarray(data).reshape(-1, 1)
        out = scaler.inverse_transform(arr)
        return out.reshape(np.asarray(data).shape)
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
        import os
        try:
            file_size = os.path.getsize(self.data_path)
            use_mmap = file_size > 100 * 1024 * 1024
            
            if use_mmap:
                arr = np.load(self.data_path, mmap_mode='r')
                print(f'Using memory mapping for large file ({file_size / 1024 / 1024:.2f} MB)')
            else:
                arr = np.load(self.data_path)
        except Exception as e:
            print(f'Warning: Failed to use memory mapping, falling back to normal load: {e}')
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
        
        # 存储原始数组引用（如果是内存映射，不会复制数据）
        self._arr = arr
        self._feature_indices = self.feature_indices
        
        # 计算分割边界
        num_samples = T
        train_end = int(num_samples * self.split[0])
        val_end = int(num_samples * (self.split[0] + self.split[1]))
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, num_samples]
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]
        
        self.scaler_params = {}
        if self.normalize:
            train_data = arr[border1s[0]:border2s[0], self.feature_indices]
            for i, feature_idx in enumerate(range(len(self.feature_indices))):
                scaler = StandardScaler()
                scaler.fit(train_data[:, i:i+1])
                self.scaler_params[feature_idx] = {
                    'mean': scaler.mean_.copy(),
                    'scale': scaler.scale_.copy()
                }
            del train_data
            import gc
            gc.collect()
        
        self.target_positions = [self.feature_indices.index(idx) if idx in self.feature_indices 
                                else None for idx in self.target_indices]
        assert None not in self.target_positions, "Some target indices are not in selected features"
        self.n_samples = ((self.border2 - self.border1) - self.seq_len) // self.stride + 1
        print(f"[{self.flag} set] Total available samples: {self.n_samples}")
        print(f"Features indices: {self.feature_indices}")
        print(f"Target indices: {self.target_indices}")

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        data_len = self.border2 - self.border1
        if s_end > data_len:
            s_begin = data_len - self.seq_len
            s_end = data_len
        
        actual_begin = self.border1 + s_begin
        actual_end = self.border1 + s_end
        
        seq = self._arr[actual_begin:actual_end, self._feature_indices].copy()
        
        if self.normalize and self.scaler_params:
            for i, feature_idx in enumerate(range(len(self._feature_indices))):
                if feature_idx in self.scaler_params:
                    params = self.scaler_params[feature_idx]
                    seq[:, i:i+1] = (seq[:, i:i+1] - params['mean']) / params['scale']
        
        seq_x = np.asarray(seq[:self.input_len], dtype=np.float32)
        seq_y = np.asarray(seq[self.input_len:, self.target_positions], dtype=np.float32)
        
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
        
        # 使用存储的参数进行逆变换
        if variable_position in self.scaler_params:
            params = self.scaler_params[variable_position]
            return data * params['scale'] + params['mean']
        return data
    
    def get_feature_indices(self):
        return self.feature_indices

    def get_target_indices(self):
        return self.target_indices


