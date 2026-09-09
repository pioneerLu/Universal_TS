import datasets
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import gc


"""
All single-variate series in UTSD are divided into (input-output) windows with a uniform length based on S3.
优化版本：使用懒加载和内存映射，避免一次性加载所有数据到内存
"""
class UTSDataset(Dataset):
    def __init__(self, dataset_path, epoch_steps, micro_bsz, 
                 flag='train', split=0.9, input_len=None, output_len=None, scale=True, stride=1):
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len + output_len
        assert flag in ['train', 'val']
        assert split >= 0 and split <= 1.0
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.scale = scale
        self.split = split
        self.stride = stride
        self.epoch_steps = epoch_steps
        self.micro_bsz = micro_bsz

        self.dataset_metadata = []
        self.n_window_list = []
        self.dataset_path = dataset_path
        self._dataset = None
        self._dataset_len = 0
        
        self.__read_metadata__()

    def __read_metadata__(self):
        """只读取元数据，不加载实际数据到内存"""
        print('Indexing dataset metadata (lazy loading)...')
        
        if self._dataset is None:
            self._dataset = datasets.load_from_disk(self.dataset_path)
            self._dataset_len = len(self._dataset)
        
        for idx, item in enumerate(tqdm(self._dataset, desc="Processing metadata")):
            data = item['target']
            data = np.array(data).reshape(-1, 1)
            num_train = int(len(data) * self.split)
            border1s = [0, num_train - self.seq_len]
            border2s = [num_train, len(data)]

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if border1 >= border2 - self.seq_len:
                continue

            scaler_params = None
            if self.scale:
                train_data = data[border1s[0]:border2s[0]]
                scaler = StandardScaler()
                scaler.fit(train_data)
                scaler_params = {
                    'mean': scaler.mean_.copy(),
                    'scale': scaler.scale_.copy()
                }
                del train_data, scaler
                gc.collect()

            data_len = border2 - border1
            n_window = (data_len - self.seq_len) // self.stride + 1
            if n_window < 1:
                continue

            self.dataset_metadata.append({
                'dataset_idx': idx,
                'border1': border1,
                'border2': border2,
                'data_len': len(data),
                'n_window': n_window,
                'scaler_params': scaler_params
            })
            prev = self.n_window_list[-1] if self.n_window_list else 0
            self.n_window_list.append(prev + n_window)

            del data

        print(f'Indexed {len(self.dataset_metadata)} time series (lazy loading enabled)')

    def _load_and_normalize_data(self, dataset_idx, border1, border2, scaler_params):
        """按需加载和归一化单个时间序列"""
        if self._dataset is None:
            self._dataset = datasets.load_from_disk(self.dataset_path)
        
        item = self._dataset[dataset_idx]
        data = np.array(item['target']).reshape(-1, 1)
        
        if self.scale and scaler_params is not None:
            data = (data - scaler_params['mean']) / scaler_params['scale']
        
        data = data[border1:border2]
        return data

    def __getitem__(self, index):
        if not self.n_window_list:
            raise IndexError("empty dataset")
        total = self.n_window_list[-1]
        index = int(index) % total
        dataset_index = 0
        while dataset_index < len(self.n_window_list) and index >= self.n_window_list[dataset_index]:
            dataset_index += 1
        if dataset_index > 0:
            index = index - self.n_window_list[dataset_index - 1]
        meta = self.dataset_metadata[dataset_index]
        data = self._load_and_normalize_data(
            meta["dataset_idx"],
            meta["border1"],
            meta["border2"],
            meta["scaler_params"],
        )
        n_timepoint = max(meta["n_window"], 1)
        s_begin = self.stride * (index % n_timepoint)
        s_end = s_begin + self.seq_len
        seq_x = np.asarray(data[s_begin:s_end], dtype=np.float32)
        seq_y = np.asarray(data[s_end:s_end + self.output_len], dtype=np.float32)
        return dict(seq_x=seq_x, seq_y=seq_y)

    def __len__(self):
        return self.epoch_steps * self.micro_bsz

class NPYDataset(Dataset):
    def __init__(self, dataset_path, epoch_steps, micro_bsz, 
                 flag='train', split=0.9, input_len=100, output_len=0, scale=False, stride=1):
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len + output_len
        assert flag in ['train', 'val']
        assert split >= 0 and split <= 1.0
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.scale = scale
        self.split = split
        self.stride = stride
        self.epoch_steps = epoch_steps
        self.micro_bsz = micro_bsz

        self.dataset_path = dataset_path
        self._data_mmap = None
        self._scaler_params = None
        
        self.__read_metadata__()

    def __read_metadata__(self):
        """使用内存映射加载数据，只计算 scaler 参数"""
        print('Loading dataset with memory mapping...')
        
        try:
            file_size = os.path.getsize(self.dataset_path)
            use_mmap = file_size > 500 * 1024 * 1024
            
            if use_mmap:
                self._data_mmap = np.load(self.dataset_path, mmap_mode='r')
                print(f'Using memory mapping for large file ({file_size / 1024 / 1024:.2f} MB)')
            else:
                self._data_mmap = np.load(self.dataset_path)
                print(f'Loading small file into memory ({file_size / 1024 / 1024:.2f} MB)')
        except Exception as e:
            print(f'Warning: Failed to use memory mapping, falling back to normal load: {e}')
            self._data_mmap = np.load(self.dataset_path)
        
        data_item = np.array(self._data_mmap).reshape(-1, 1)
        
        num_train = int(len(data_item) * self.split)
        border1s = [0, num_train - self.seq_len]
        border2s = [num_train, len(data_item)]

        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        # 计算并存储 scaler 参数
        if self.scale:
            train_data = data_item[border1s[0]:border2s[0]]
            scaler = StandardScaler()
            scaler.fit(train_data)
            self._scaler_params = {
                'mean': scaler.mean_.copy(),
                'scale': scaler.scale_.copy()
            }
            del train_data, scaler
            gc.collect()

        # 计算可用窗口数量
        data_len = self.border2 - self.border1
        self.n_window = (data_len - self.seq_len) // self.stride + 1
        
        print(f'Dataset indexed: {self.n_window} windows available')

    def _get_normalized_data(self):
        """按需获取归一化后的数据片段"""
        if isinstance(self._data_mmap, np.memmap):
            data = np.array(self._data_mmap).reshape(-1, 1)
        else:
            data = self._data_mmap.reshape(-1, 1)
        
        if self.scale and self._scaler_params is not None:
            data = (data - self._scaler_params['mean']) / self._scaler_params['scale']
        
        data = data[self.border1:self.border2]
        return data

    def __getitem__(self, index):
        data = self._get_normalized_data()
        n_timepoint = max(self.n_window, 1)
        s_begin = self.stride * (int(index) % n_timepoint)
        s_end = s_begin + self.seq_len
        seq_x = np.asarray(data[s_begin:s_end], dtype=np.float32)
        seq_y = np.asarray(data[s_end:s_end + self.output_len], dtype=np.float32)
        return dict(seq_x=seq_x, seq_y=seq_y)

    def __len__(self):
        return self.epoch_steps * self.micro_bsz
    

# See ```download_dataset.py``` to download the dataset first
if __name__ == '__main__':
    dataset = UTSDataset(dataset_path="data/pretrain/utsd", 
                         input_len=720, output_len=96, flag='train')
    print(f'total {len(dataset)} time series windows (sentence)')
    item = dataset[0]
    print(item['seq_x'].shape, item['seq_y'].shape)