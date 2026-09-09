#!/usr/bin/env python3
"""
数据预处理脚本 - 单卡处理单变量npy数据并生成缓存

用法:
    python preprocess_data.py \
        --data_file path/to/data_dir \
        --cache_dir path/to/cache_dir \
        --input_len 816 \
        --output_len 0 \
        --split 0.8 \
        --norm True \
        --stride 1 \
        --merge_interval 50 \
        --async_merge True
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import gc
import threading
from queue import Queue
import shutil


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess npy dataset and generate cache')
    parser.add_argument("--data_file", type=str, required=True, help="Directory containing npy files")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory to save preprocessed data")
    parser.add_argument("--input_len", type=int, required=True, help="Input sequence length")
    parser.add_argument("--output_len", type=int, default=1, help="Output sequence length")
    parser.add_argument("--split", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--norm", type=str2bool, default=True, help="Whether to normalize data")
    parser.add_argument("--stride", type=int, default=1, help="Stride for window generation")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for writing windows to disk")
    parser.add_argument("--merge_interval", type=int, default=100, help="Number of batches before merging into a chunk")
    parser.add_argument("--async_merge", type=str2bool, default=True, help="Use async merge in background thread")
    return parser.parse_args()


def check_cache_exists(cache_dir):
    chunks_info_path = os.path.join(cache_dir, 'chunks_info.json')
    if os.path.exists(chunks_info_path):
        chunks_data = json.load(open(chunks_info_path, 'r'))
        if chunks_data.get('total_train', 0) > 0 and chunks_data.get('total_val', 0) > 0:
            print(f"Chunked cache already exists at {cache_dir}")
            return True
    
    cache_info_path = os.path.join(cache_dir, 'cache_info.json')
    train_cache_path = os.path.join(cache_dir, 'train_windows.npy')
    val_cache_path = os.path.join(cache_dir, 'val_windows.npy')
    
    if all(os.path.exists(p) for p in [cache_info_path, train_cache_path, val_cache_path]):
        print(f"Cache already exists at {cache_dir}")
        return True
    return False


def load_progress(cache_dir):
    progress_path = os.path.join(cache_dir, 'progress.json')
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def save_progress(cache_dir, progress):
    progress_path = os.path.join(cache_dir, 'progress.json')
    progress_serializable = convert_to_json_serializable(progress)
    with open(progress_path, 'w') as f:
        json.dump(progress_serializable, f, indent=2)


def get_existing_batches(temp_dir):
    train_batches = []
    val_batches = []
    
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            if f.startswith('train_batch_') and f.endswith('.npy'):
                try:
                    idx = int(f.replace('train_batch_', '').replace('.npy', ''))
                    train_batches.append((idx, os.path.join(temp_dir, f)))
                except:
                    pass
            elif f.startswith('val_batch_') and f.endswith('.npy'):
                try:
                    idx = int(f.replace('val_batch_', '').replace('.npy', ''))
                    val_batches.append((idx, os.path.join(temp_dir, f)))
                except:
                    pass
    
    train_batches.sort(key=lambda x: x[0])
    val_batches.sort(key=lambda x: x[0])
    return train_batches, val_batches


def get_existing_chunks(cache_dir):
    chunks_info = []
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            if f.startswith('train_windows_chunk_') and f.endswith('.npy'):
                try:
                    idx = int(f.replace('train_windows_chunk_', '').replace('.npy', ''))
                    chunks_info.append({
                        'chunk_idx': idx,
                        'type': 'train',
                        'file': f,
                        'num_windows': np.load(os.path.join(cache_dir, f), mmap_mode='r').shape[0]
                    })
                except:
                    pass
            elif f.startswith('val_windows_chunk_') and f.endswith('.npy'):
                try:
                    idx = int(f.replace('val_windows_chunk_', '').replace('.npy', ''))
                    chunks_info.append({
                        'chunk_idx': idx,
                        'type': 'val',
                        'file': f,
                        'num_windows': np.load(os.path.join(cache_dir, f), mmap_mode='r').shape[0]
                    })
                except:
                    pass
    return chunks_info


def merge_batches_to_chunk(batch_files, chunk_path, temp_dir, chunk_type="train"):
    if not batch_files:
        return 0
    
    first_batch = np.load(batch_files[0], mmap_mode='r')
    batch_shape = first_batch.shape
    window_shape = batch_shape[1:] if len(batch_shape) > 1 else ()
    del first_batch
    
    total_size = 0
    batch_sizes = []
    for f in batch_files:
        if os.path.exists(f):
            batch_mmap = np.load(f, mmap_mode='r')
            size = batch_mmap.shape[0]
            batch_sizes.append(size)
            total_size += size
            del batch_mmap
    
    if total_size == 0:
        return 0
    
    final_shape = (total_size,) + window_shape
    temp_final = os.path.join(temp_dir, f'{chunk_type}_chunk_temp.dat')
    chunk_array = np.memmap(temp_final, mode='w+', dtype=np.float32, shape=final_shape)
    
    offset = 0
    for i, batch_file in enumerate(batch_files):
        if os.path.exists(batch_file):
            batch_data = np.load(batch_file, mmap_mode='r')
            batch_size = batch_sizes[i]
            chunk_array[offset:offset+batch_size] = batch_data[:]
            offset += batch_size
            del batch_data
    
    chunk_array.flush()
    del chunk_array
    gc.collect()
    
    final_array = np.memmap(temp_final, mode='r', dtype=np.float32, shape=final_shape)
    np.save(chunk_path, final_array)
    del final_array
    if os.path.exists(temp_final):
        os.remove(temp_final)
    gc.collect()
    
    return total_size


def merge_worker(merge_queue, chunks_info, cache_dir, temp_dir, lock):
    while True:
        task = merge_queue.get()
        if task is None:
            break
        
        chunk_type, chunk_idx, batch_files, start_batch, end_batch = task
        
        try:
            chunk_path = os.path.join(cache_dir, f'{chunk_type}_windows_chunk_{chunk_idx:03d}.npy')
            num_windows = merge_batches_to_chunk(batch_files, chunk_path, temp_dir, chunk_type)
            
            for f in batch_files:
                if os.path.exists(f):
                    os.remove(f)
            
            chunk_info = {
                'chunk_idx': chunk_idx,
                'type': chunk_type,
                'file': f'{chunk_type}_windows_chunk_{chunk_idx:03d}.npy',
                'num_windows': num_windows,
                'batch_range': (start_batch, end_batch)
            }
            
            with lock:
                existing = next((c for c in chunks_info if c.get('chunk_idx') == chunk_idx and c.get('type') == chunk_type), None)
                if existing:
                    existing.update(chunk_info)
                else:
                    chunks_info.append(chunk_info)
            
            print(f"\nMerged {chunk_type} chunk {chunk_idx}: {num_windows:,} windows")
        except Exception as e:
            print(f"\nError merging {chunk_type} chunk {chunk_idx}: {e}")
        finally:
            merge_queue.task_done()


def preprocess_npy_dataset(data_file, cache_dir, input_len, output_len, split, norm, stride, batch_size=10000, merge_interval=100, async_merge=True):
    os.makedirs(data_file, exist_ok=True)
    
    if check_cache_exists(cache_dir):
        return
    
    os.makedirs(cache_dir, exist_ok=True)
    temp_dir = os.path.join(cache_dir, 'temp_batches')
    os.makedirs(temp_dir, exist_ok=True)
    
    seq_len = input_len + output_len
    
    print(f'Loading NPY files from {data_file}...')
    npy_files = [f for f in os.listdir(data_file) if f.endswith('.npy')]
    npy_files.sort()
    
    if not npy_files:
        raise ValueError(f"No npy files found in {data_file}")
    
    print(f"Found {len(npy_files)} npy files")
    print(f"Using batch size: {batch_size} windows per batch")
    if merge_interval > 0:
        print(f"Merge interval: {merge_interval} batches per chunk")
        print(f"Async merge: {async_merge}")
    
    progress = load_progress(cache_dir)
    existing_chunks = get_existing_chunks(cache_dir)
    train_batches, val_batches = get_existing_batches(temp_dir)
    
    if progress is not None:
        print(f"\n Found previous progress: processed {progress.get('processed_files', 0)}/{len(npy_files)} files")
        print(f"  Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")
        print(f"  Existing chunks: {len(existing_chunks)}")
        print(f"  Resuming from file: {progress.get('last_file', 'unknown')}")
        
        processed_files = set(progress.get('processed_files_list', []))
        start_file_idx = progress.get('last_file_idx', 0)
        train_batch_idx = progress.get('train_batch_idx', len(train_batches))
        val_batch_idx = progress.get('val_batch_idx', len(val_batches))
        total_train_windows = progress.get('total_train_windows', 0)
        total_val_windows = progress.get('total_val_windows', 0)
        scaler_params_list = progress.get('scaler_params_list', [])
        chunk_idx = progress.get('chunk_idx', len([c for c in existing_chunks if c['type'] == 'train']))
        chunks_info = existing_chunks + progress.get('chunks_info', [])
        
        npy_files = [f for f in npy_files if f not in processed_files]
        if not npy_files:
            print("All files already processed, proceeding to final merge...")
    else:
        start_file_idx = 0
        train_batch_idx = len(train_batches)
        val_batch_idx = len(val_batches)
        total_train_windows = sum(np.load(f[1], mmap_mode='r').shape[0] for f in train_batches) if train_batches else 0
        total_val_windows = sum(np.load(f[1], mmap_mode='r').shape[0] for f in val_batches) if val_batches else 0
        scaler_params_list = []
        processed_files = set()
        chunk_idx = len([c for c in existing_chunks if c['type'] == 'train'])
        chunks_info = existing_chunks
    
    train_windows_batch = []
    val_windows_batch = []
    chunks_info_lock = threading.Lock()
    
    merge_queue = None
    merge_thread = None
    if merge_interval > 0 and async_merge:
        merge_queue = Queue()
        merge_thread = threading.Thread(target=merge_worker, args=(merge_queue, chunks_info, cache_dir, temp_dir, chunks_info_lock), daemon=True)
        merge_thread.start()
    
    if npy_files:
        pbar_files = tqdm(npy_files, desc="Processing files", unit="file", initial=start_file_idx, total=len(npy_files) + start_file_idx)
    else:
        pbar_files = None
    
    for file_idx, file_name in enumerate(npy_files):
        file_path = os.path.join(data_file, file_name)
        
        if pbar_files:
            pbar_files.set_postfix({
            'train_windows': total_train_windows,
            'val_windows': total_val_windows,
            'file': file_name[:30] + '...' if len(file_name) > 30 else file_name
        })
        
        try:
            file_size = os.path.getsize(file_path)
            use_mmap = file_size > 500 * 1024 * 1024
            raw_data = np.load(file_path, mmap_mode='r' if use_mmap else None, allow_pickle=True)
        except Exception as e:
            tqdm.write(f'Warning: Failed to load {file_name}: {e}')
            continue
        
        num_items = raw_data.shape[0]
        for item_idx in range(num_items):
            ts_data = np.array(raw_data[item_idx]).reshape(-1, 1)
            
            if len(ts_data) < seq_len:
                continue
            
            num_train = int(len(ts_data) * split)
            border1s = [0, num_train - seq_len]
            border2s = [num_train, len(ts_data)]
            
            if border1s[1] >= border2s[1] - seq_len:
                continue
            
            scaler_params = None
            if norm:
                scaler = StandardScaler()
                train_data = ts_data[border1s[0]:border2s[0]]
                scaler.fit(train_data)
                scaler_params = {
                    'mean': scaler.mean_.copy(),
                    'scale': scaler.scale_.copy()
                }
                scaler_params_list.append({
                    'file_name': file_name,
                    'item_idx': item_idx,
                    'params': scaler_params
                })
                ts_data = (ts_data - scaler_params['mean']) / scaler_params['scale']
                del train_data, scaler
            
            train_data = ts_data[border1s[0]:border2s[0]]
            train_len = len(train_data)
            train_n_window = (train_len - seq_len) // stride + 1
            
            if train_n_window > 0:
                for s_begin in range(0, train_len - seq_len + 1, stride):
                    s_end = s_begin + seq_len
                    window = train_data[s_begin:s_end].copy()
                    train_windows_batch.append(window)
                    total_train_windows += 1
                    
                    if len(train_windows_batch) >= batch_size:
                        batch_array = np.array(train_windows_batch, dtype=np.float32)
                        batch_file = os.path.join(temp_dir, f'train_batch_{train_batch_idx:06d}.npy')
                        np.save(batch_file, batch_array)
                        train_windows_batch = []
                        train_batch_idx += 1
                        del batch_array
                        gc.collect()
                        
                        if merge_interval > 0 and train_batch_idx % merge_interval == 0:
                            start_batch = train_batch_idx - merge_interval
                            train_batch_files = [os.path.join(temp_dir, f'train_batch_{i:06d}.npy') for i in range(start_batch, train_batch_idx)]
                            
                            if async_merge and merge_queue is not None:
                                merge_queue.put(('train', chunk_idx, train_batch_files, start_batch, train_batch_idx))
                            else:
                                chunk_path = os.path.join(cache_dir, f'train_windows_chunk_{chunk_idx:03d}.npy')
                                num_windows = merge_batches_to_chunk(train_batch_files, chunk_path, temp_dir, "train")
                                
                                for f in train_batch_files:
                                    if os.path.exists(f):
                                        os.remove(f)
                                
                                chunks_info.append({
                                    'chunk_idx': chunk_idx,
                                    'type': 'train',
                                    'file': f'train_windows_chunk_{chunk_idx:03d}.npy',
                                    'num_windows': num_windows,
                                    'batch_range': (start_batch, train_batch_idx)
                                })
                                print(f"\n✓ Merged train chunk {chunk_idx}: {num_windows:,} windows")
                            
                            chunk_idx += 1
                    
                    if total_train_windows % 1000 == 0 and pbar_files:
                        pbar_files.set_postfix({
                            'train_windows': total_train_windows,
                            'val_windows': total_val_windows,
                            'train_batches': train_batch_idx,
                            'val_batches': val_batch_idx,
                            'file': file_name[:30] + '...' if len(file_name) > 30 else file_name
                        })
            
            val_data = ts_data[border1s[1]:border2s[1]]
            val_len = len(val_data)
            val_n_window = (val_len - seq_len) // stride + 1
            
            if val_n_window > 0:
                for s_begin in range(0, val_len - seq_len + 1, stride):
                    s_end = s_begin + seq_len
                    window = val_data[s_begin:s_end].copy()
                    val_windows_batch.append(window)
                    total_val_windows += 1
                    
                    if len(val_windows_batch) >= batch_size:
                        batch_array = np.array(val_windows_batch, dtype=np.float32)
                        batch_file = os.path.join(temp_dir, f'val_batch_{val_batch_idx:06d}.npy')
                        np.save(batch_file, batch_array)
                        val_windows_batch = []
                        val_batch_idx += 1
                        del batch_array
                        gc.collect()
                        
                        if merge_interval > 0 and val_batch_idx % merge_interval == 0:
                            val_start_batch = val_batch_idx - merge_interval
                            val_batch_files = [os.path.join(temp_dir, f'val_batch_{i:06d}.npy') for i in range(val_start_batch, val_batch_idx) if os.path.exists(os.path.join(temp_dir, f'val_batch_{i:06d}.npy'))]
                            if val_batch_files:
                                val_chunk_idx = (val_batch_idx // merge_interval) - 1
                                
                                if async_merge and merge_queue is not None:
                                    merge_queue.put(('val', val_chunk_idx, val_batch_files, val_start_batch, val_batch_idx))
                                else:
                                    val_chunk_path = os.path.join(cache_dir, f'val_windows_chunk_{val_chunk_idx:03d}.npy')
                                    val_num_windows = merge_batches_to_chunk(val_batch_files, val_chunk_path, temp_dir, "val")
                                    
                                    for f in val_batch_files:
                                        if os.path.exists(f):
                                            os.remove(f)
                                    
                                    val_chunk_info = next((c for c in chunks_info if c.get('chunk_idx') == val_chunk_idx and c.get('type') == 'val'), None)
                                    if val_chunk_info:
                                        val_chunk_info['num_windows'] = val_num_windows
                                        val_chunk_info['batch_range'] = (val_start_batch, val_batch_idx)
                                    else:
                                        chunks_info.append({
                                            'chunk_idx': val_chunk_idx,
                                            'type': 'val',
                                            'file': f'val_windows_chunk_{val_chunk_idx:03d}.npy',
                                            'num_windows': val_num_windows,
                                            'batch_range': (val_start_batch, val_batch_idx)
                                        })
                                    print(f"✓ Merged val chunk {val_chunk_idx}: {val_num_windows:,} windows")
                    
                    if total_val_windows % 1000 == 0 and pbar_files:
                        pbar_files.set_postfix({
                            'train_windows': total_train_windows,
                            'val_windows': total_val_windows,
                            'train_batches': train_batch_idx,
                            'val_batches': val_batch_idx,
                            'file': file_name[:30] + '...' if len(file_name) > 30 else file_name
                        })
            
            del ts_data, train_data, val_data
            gc.collect()
        
        del raw_data
        gc.collect()
    
        processed_files.add(file_name)
        
        if (file_idx + 1) % 5 == 0 or file_idx == len(npy_files) - 1:
            save_progress(cache_dir, {
                'processed_files': len(processed_files),
                'processed_files_list': list(processed_files),
                'last_file': file_name,
                'last_file_idx': start_file_idx + file_idx + 1,
                'train_batch_idx': train_batch_idx,
                'val_batch_idx': val_batch_idx,
                'total_train_windows': total_train_windows,
                'total_val_windows': total_val_windows,
                'scaler_params_list': scaler_params_list,
                'chunk_idx': chunk_idx,
                'chunks_info': [c for c in chunks_info if 'batch_range' not in c or c.get('batch_range', (0, 0))[1] <= train_batch_idx],
                'input_len': input_len,
                'output_len': output_len,
                'split': split,
                'norm': norm,
                'stride': stride,
                'merge_interval': merge_interval
            })
    
    if len(train_windows_batch) > 0:
        batch_array = np.array(train_windows_batch, dtype=np.float32)
        batch_file = os.path.join(temp_dir, f'train_batch_{train_batch_idx:06d}.npy')
        np.save(batch_file, batch_array)
        train_batch_idx += 1
        del train_windows_batch, batch_array
        gc.collect()
    
    if len(val_windows_batch) > 0:
        batch_array = np.array(val_windows_batch, dtype=np.float32)
        batch_file = os.path.join(temp_dir, f'val_batch_{val_batch_idx:06d}.npy')
        np.save(batch_file, batch_array)
        val_batch_idx += 1
        del val_windows_batch, batch_array
        gc.collect()
    
    if pbar_files:
        pbar_files.close()
        print(f"\n✓ Generated {total_train_windows:,} train windows ({train_batch_idx} batches) and {total_val_windows:,} val windows ({val_batch_idx} batches)")
        
    if merge_interval > 0:
        last_train_chunk_start = (train_batch_idx // merge_interval) * merge_interval
        last_val_chunk_start = (val_batch_idx // merge_interval) * merge_interval
        
        remaining_train_batches = list(range(last_train_chunk_start, train_batch_idx)) if last_train_chunk_start < train_batch_idx else []
        remaining_val_batches = list(range(last_val_chunk_start, val_batch_idx)) if last_val_chunk_start < val_batch_idx else []
        
        if remaining_train_batches:
            print("\nMerging remaining batches...")
            train_batch_files = [os.path.join(temp_dir, f'train_batch_{i:06d}.npy') for i in remaining_train_batches if os.path.exists(os.path.join(temp_dir, f'train_batch_{i:06d}.npy'))]
            if train_batch_files:
                if async_merge and merge_queue is not None:
                    merge_queue.put(('train', chunk_idx, train_batch_files, remaining_train_batches[0], train_batch_idx))
                else:
                    chunk_path = os.path.join(cache_dir, f'train_windows_chunk_{chunk_idx:03d}.npy')
                    num_windows = merge_batches_to_chunk(train_batch_files, chunk_path, temp_dir, "train")
                    
                    for f in train_batch_files:
                        if os.path.exists(f):
                            os.remove(f)
                    
                    chunks_info.append({
                        'chunk_idx': chunk_idx,
                        'type': 'train',
                        'file': f'train_windows_chunk_{chunk_idx:03d}.npy',
                        'num_windows': num_windows,
                        'batch_range': (remaining_train_batches[0], train_batch_idx)
                    })
                    print(f"✓ Merged final train chunk {chunk_idx}: {num_windows:,} windows")
        
        if remaining_val_batches:
            val_batch_files = [os.path.join(temp_dir, f'val_batch_{i:06d}.npy') for i in remaining_val_batches if os.path.exists(os.path.join(temp_dir, f'val_batch_{i:06d}.npy'))]
            if val_batch_files:
                val_chunk_idx = val_batch_idx // merge_interval
                
                if async_merge and merge_queue is not None:
                    merge_queue.put(('val', val_chunk_idx, val_batch_files, remaining_val_batches[0], val_batch_idx))
                else:
                    val_chunk_path = os.path.join(cache_dir, f'val_windows_chunk_{val_chunk_idx:03d}.npy')
                    val_num_windows = merge_batches_to_chunk(val_batch_files, val_chunk_path, temp_dir, "val")
                    
                    for f in val_batch_files:
                        if os.path.exists(f):
                            os.remove(f)
                    
                    val_chunk_info = next((c for c in chunks_info if c.get('chunk_idx') == val_chunk_idx and c.get('type') == 'val'), None)
                    if val_chunk_info:
                        val_chunk_info['num_windows'] = val_num_windows
                        val_chunk_info['batch_range'] = (remaining_val_batches[0], val_batch_idx)
                    else:
                        chunks_info.append({
                            'chunk_idx': val_chunk_idx,
                            'type': 'val',
                            'file': f'val_windows_chunk_{val_chunk_idx:03d}.npy',
                            'num_windows': val_num_windows,
                            'batch_range': (remaining_val_batches[0], val_batch_idx)
                        })
                    print(f"✓ Merged final val chunk {val_chunk_idx}: {val_num_windows:,} windows")
    
    save_progress(cache_dir, {
        'processed_files': len(processed_files),
        'processed_files_list': list(processed_files),
        'last_file': npy_files[-1] if npy_files else '',
        'last_file_idx': start_file_idx + len(npy_files),
        'train_batch_idx': train_batch_idx,
        'val_batch_idx': val_batch_idx,
        'total_train_windows': total_train_windows,
        'total_val_windows': total_val_windows,
        'scaler_params_list': scaler_params_list,
        'chunk_idx': chunk_idx,
        'chunks_info': chunks_info,
        'input_len': input_len,
        'output_len': output_len,
        'split': split,
        'norm': norm,
        'stride': stride,
        'merge_interval': merge_interval,
        'merging': True
    })
    
    if merge_queue is not None:
        merge_queue.join()
        merge_queue.put(None)
        merge_thread.join()
        print("✓ All async merge tasks completed")
    
    if merge_interval == 0:
        print("\nMerging all batch files into single files...")
        train_cache_path = os.path.join(cache_dir, 'train_windows.npy')
        val_cache_path = os.path.join(cache_dir, 'val_windows.npy')
        
        if train_batch_idx > 0:
            print(f"Merging {train_batch_idx} train batches...")
            train_batch_files = [os.path.join(temp_dir, f'train_batch_{i:06d}.npy') for i in range(train_batch_idx)]
            
            first_batch = np.load(train_batch_files[0], mmap_mode='r')
            batch_shape = first_batch.shape
            window_shape = batch_shape[1:] if len(batch_shape) > 1 else ()
            del first_batch
            
            total_size = sum(np.load(f, mmap_mode='r').shape[0] for f in train_batch_files)
            final_shape = (total_size,) + window_shape
            
            temp_final = os.path.join(temp_dir, 'train_final_temp.dat')
            train_array = np.memmap(temp_final, mode='w+', dtype=np.float32, shape=final_shape)
            
            offset = 0
            with tqdm(total=train_batch_idx, desc="Merging train batches", unit="batch") as pbar:
                for batch_file in train_batch_files:
                    batch_data = np.load(batch_file, mmap_mode='r')
                    batch_size = batch_data.shape[0]
                    train_array[offset:offset+batch_size] = batch_data[:]
                    offset += batch_size
                    pbar.update(1)
                    del batch_data
                    gc.collect()
            
            train_array.flush()
            del train_array
            gc.collect()
            
            final_array = np.memmap(temp_final, mode='r', dtype=np.float32, shape=final_shape)
            np.save(train_cache_path, final_array)
            del final_array
            os.remove(temp_final)
            gc.collect()
            print(f"✓ Train windows shape: {final_shape}")
        
        if val_batch_idx > 0:
            print(f"Merging {val_batch_idx} val batches...")
            val_batch_files = [os.path.join(temp_dir, f'val_batch_{i:06d}.npy') for i in range(val_batch_idx)]
            
            first_batch = np.load(val_batch_files[0], mmap_mode='r')
            batch_shape = first_batch.shape
            window_shape = batch_shape[1:] if len(batch_shape) > 1 else ()
            del first_batch
            
            total_size = sum(np.load(f, mmap_mode='r').shape[0] for f in val_batch_files)
            final_shape = (total_size,) + window_shape
            
            temp_final = os.path.join(temp_dir, 'val_final_temp.dat')
            val_array = np.memmap(temp_final, mode='w+', dtype=np.float32, shape=final_shape)
            
            offset = 0
            with tqdm(total=val_batch_idx, desc="Merging val batches", unit="batch") as pbar:
                for batch_file in val_batch_files:
                    batch_data = np.load(batch_file, mmap_mode='r')
                    batch_size = batch_data.shape[0]
                    val_array[offset:offset+batch_size] = batch_data[:]
                    offset += batch_size
                    pbar.update(1)
                    del batch_data
                    gc.collect()
            
            val_array.flush()
            del val_array
            gc.collect()
            
            final_array = np.memmap(temp_final, mode='r', dtype=np.float32, shape=final_shape)
            np.save(val_cache_path, final_array)
            del final_array
            os.remove(temp_final)
            gc.collect()
            print(f"✓ Val windows shape: {final_shape}")
    
    print("\nCleaning up temporary batch files...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print("Temporary files cleaned up")
    
    if norm and scaler_params_list:
        scaler_cache_path = os.path.join(cache_dir, 'scaler_params.npy')
        np.save(scaler_cache_path, scaler_params_list, allow_pickle=True)
    
    if merge_interval > 0:
        train_chunks = [c for c in chunks_info if c['type'] == 'train']
        val_chunks = [c for c in chunks_info if c['type'] == 'val']
        total_train_chunks = sum(c['num_windows'] for c in train_chunks)
        total_val_chunks = sum(c['num_windows'] for c in val_chunks)
        
        chunks_info_path = os.path.join(cache_dir, 'chunks_info.json')
        chunks_data = {
            "input_len": input_len,
            "output_len": output_len,
            "seq_len": seq_len,
            "split": split,
            "norm": norm,
            "stride": stride,
            "merge_interval": merge_interval,
            "chunks": chunks_info,
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
            "total_train": int(total_train_chunks),
            "total_val": int(total_val_chunks),
            "data_file": os.path.abspath(data_file)
        }
        
        with open(chunks_info_path, 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        cache_info = {
            "input_len": input_len,
            "output_len": output_len,
            "seq_len": seq_len,
            "split": split,
            "norm": norm,
            "stride": stride,
            "num_train": int(total_train_chunks),
            "num_val": int(total_val_chunks),
            "data_file": os.path.abspath(data_file)
        }
        
        progress_path = os.path.join(cache_dir, 'progress.json')
        if os.path.exists(progress_path):
            os.remove(progress_path)
            print("Progress file cleaned up")
        
        print(f"\n Cache saved to {cache_dir}")
        print(f"Generated {len(train_chunks)} train chunks ({total_train_chunks:,} windows)")
        print(f"Generated {len(val_chunks)} val chunks ({total_val_chunks:,} windows)")
        print(f"Chunks info saved to chunks_info.json")
    else:
        cache_info = {
        "input_len": input_len,
        "output_len": output_len,
        "seq_len": seq_len,
        "split": split,
        "norm": norm,
        "stride": stride,
        "num_train": int(total_train_windows),
        "num_val": int(total_val_windows),
        "data_file": os.path.abspath(data_file)
    }
    
    cache_info_path = os.path.join(cache_dir, 'cache_info.json')
    with open(cache_info_path, 'w') as f:
        json.dump(cache_info, f, indent=2)
    
    print(f"\n✓ Cache saved to {cache_dir}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Data Preprocessing for Cache Generation")
    print("=" * 60)
    print(f"Data directory: {args.data_file}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Input length: {args.input_len}")
    print(f"Output length: {args.output_len}")
    print(f"Split ratio: {args.split}")
    print(f"Normalize: {args.norm}")
    print(f"Stride: {args.stride}")
    print(f"Batch size: {args.batch_size}")
    print(f"Merge interval: {args.merge_interval} batches per chunk" if args.merge_interval > 0 else "Merge interval: disabled (merge all at end)")
    if args.merge_interval > 0:
        print(f"Async merge: {args.async_merge}")
    print("=" * 60)
    
    preprocess_npy_dataset(
        data_file=args.data_file,
        cache_dir=args.cache_dir,
        input_len=args.input_len,
        output_len=args.output_len,
        split=args.split,
        norm=args.norm,
        stride=args.stride,
        batch_size=args.batch_size,
        merge_interval=args.merge_interval,
        async_merge=args.async_merge
    )
    
    print("\nPreprocessing completed!")


if __name__ == '__main__':
    main()
