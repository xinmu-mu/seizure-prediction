import pyedflib
import numpy as np
import os
from scipy.signal import butter, filtfilt
from scipy.signal import resample
import pandas as pd
from collections import defaultdict


#I.edf数据读取

root_folder_path = 'chb'
def get_files_in_subfolders(root_folder):
    files_dict = {}

    for root, dirs, files in os.walk(root_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name in files_dict:
                files_dict[file_name].append(file_path)
            else:
                files_dict[file_name] = [file_path]

    return files_dict


def read_edf_signals(files_in_subfolders, target_channels):
    signals_dict = {}

    for file_name, file_paths in files_in_subfolders.items():
        file_path = file_paths[0]
        file_path = file_path.replace('\\', '/')

        # 检查文件名是否以 .edf 结尾并且包含目标通道之一
        if file_name.lower().endswith('.edf') and any(channel in file_name.lower() for channel in target_channels):
            try:
                with pyedflib.EdfReader(file_path) as f:
                    n = f.signals_in_file
                    sample_lengths = [f.getNSamples()[i] for i in range(n)]
                    num_samples = sample_lengths[0]
                    signals = np.zeros((n, num_samples))
                    for i in range(n):
                        signal = f.readSignal(i)
                        signals[i, :] = signal
                    signals_dict[file_name] = signals
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    return signals_dict



#II.预处理

#滤波
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=1)
    return y

def process_patient_data(patient_data,lowcut,highcut,fs):
    filtered_patient_data = {}

    for file_name, tuples_list in patient_data.items():
        filtered_list = []

        for data_tuple in tuples_list:
            array, label = data_tuple

            filtered_array = butter_bandpass_filter(array,lowcut,highcut,fs)
            filtered_list.append((filtered_array, label))

        filtered_patient_data[file_name] = filtered_list

    return filtered_patient_data



#降采样
Fs = 400
target_Fs = 256
def downsample_data(Fs, target_Fs,filtered_patient_data):
    ratio = target_Fs / Fs  # 正确的降采样比例
    downsampled_data = []
    for sublist in filtered_patient_data:
        downsampled_sublist = []
        for array in sublist:
            if array.ndim == 1:
                array = array.reshape(1, -1)

            original_samples = array.shape[-1]
            target_samples = int(original_samples * ratio)

            resampled = resample(array, target_samples, axis=-1)
            downsampled_sublist.append(resampled)

        concatenated_array = np.concatenate(downsampled_sublist, axis=0)
        downsampled_data.append(concatenated_array)
    return downsampled_data

#标准化
def z_score_normalize(siganls):
    scale_signal =[]
    for signal_data in siganls:
        mean = np.mean(signal_data, axis=1, keepdims=True)
        std = np.std(signal_data, axis=1, keepdims=True)
        normalized_data = (signal_data - mean) / std
        scale_signal.append(normalized_data)

    return scale_signal




#III.数据分段

#读取癫痫发作时间
def get_files_seizures(files_in_subfolders):
    seizure_events = []
    for file_name, file_paths in files_in_subfolders.items():
        file_path = file_paths[0]
        file_path = file_path.replace('\\', '/')
        if file_name.lower().endswith('.csv') and 'CHB-MIT DB timestamp' in file_name.lower():
            sz = pd.read_csv(file_path)
            seizure_events = []
            for index, row in sz.iterrows():
                event = row['Sub File']
                start_time = row['Seizure Start Time']
                end_time = row['Seizure End Time']
                seizure_events.append([event,start_time, end_time])
    return seizure_events



#划分前期，间期，发作期
def makeEEGClips(freq_samp, signals, seiz_time , ictal, interictal ,preictal):
    preictal_dict = {}
    interictal_dict = {}
    ictal_dict = {}
    for seiz in seiz_time:
        file_name, seiz_start, seiz_end = seiz
        file_name = file_name + '.edf'
        if file_name in signals:
            signal_data = signals[file_name]
            ictal_start = seiz_start * freq_samp
            ictal_end = seiz_end * freq_samp
            ictal_data = signal_data[:,ictal_start:ictal_end]
            if file_name not in ictal_dict:
                ictal_dict[file_name] = []
            ictal_dict[file_name].append(ictal_data)
            if seiz_start > ictal+preictal:
                preictal_start = int((seiz_start-preictal-ictal)*freq_samp)
                preictal_end = int((seiz_start-ictal)*freq_samp)
                interictal_start = preictal_end
                interictal_end = int((seiz_start-ictal+interictal)*freq_samp)
                interictal_end = min(interictal_end, signal_data.shape[1])
                preictal_data = signal_data[:,preictal_start:preictal_end]
                interictal_data = signal_data[:,interictal_start:interictal_end]
                if file_name not in preictal_dict:
                    preictal_dict[file_name] = []
                preictal_dict[file_name].append(preictal_data)
                if file_name not in interictal_dict:
                    interictal_dict[file_name] = []
                interictal_dict[file_name].append(interictal_data)

    return preictal_dict, interictal_dict, ictal_dict




#分段
def slice_data(eeg_data, slice_samples):
    signal_clip = {}

    for file_name, data_list in eeg_data.items():
        slices = []
        for data in data_list:
            num_clip = data.shape[1]//slice_samples
            for start in range(num_clip):
                start_step = start * slice_samples
                end_step = start_step + slice_samples
                slice_data = data[:, start_step:end_step]
                slices.append(slice_data)
        signal_clip[file_name] = slices
    return signal_clip


#IV.数据集划分（留一法）
def split_data_by_patient(patient_data, test_patient_name):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for patient_name, samples in patient_data.items():
        if patient_name == test_patient_name:
            # 当前病人为测试病人
            for sample in samples:
                array, label = sample

                if array.shape[0] > 22:
                    array = array[:22, :]
                test_data.append(array)
                test_labels.append(label)
        else:
            # 其他病人为训练病人
            for sample in samples:
                array, label = sample

                if array.shape[0] > 22:
                    array = array[:22, :]
                train_data.append(array)
                train_labels.append(label)

    try:
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)
    except ValueError as e:
        print(f"Error converting to numpy array: {e}")
        raise

    return train_data, train_labels, test_data, test_labels


def make_patientdata(preictal_clip, interictal_clip):
    data = []
    labels = []
    patient_data = defaultdict(list)
    # 处理 interictal_clip
    for file_name, arrays in interictal_clip.items():
        patient_name = file_name.split('-')[0]
        for array in arrays:
            data.append(array)
            labels.append(1)  # 间期数据标签为 1
            patient_data[patient_name].append((array, 1))
    # 处理 preictal_clip
    for file_name, arrays in preictal_clip.items():
        patient_name = file_name.split('-')[0]
        for array in arrays:
            data.append(array)
            labels.append(0)  # 前期数据标签为 0
            patient_data[patient_name].append((array, 0))
    return patient_data


def main_process():
    # 设置参数
    root_folder = 'chb'
    target_channels = ['chb14', 'chb16', 'chb17', 'chb18', 'chb19', 'chb20', 'chb21', 'chb22', 'chb23']
    lowcut, highcut = 0.5, 50  # 滤波频带
    Fs = 256  # 采样率
    slice_samples = 1280  # 5秒窗口
    preictal, ictal, interictal = 600, 90, 1200  # 时间参数

    # 获取所有文件路径
    files_dict = get_files_in_subfolders(root_folder)

    # 读取EDF信号
    signals_dict = read_edf_signals(files_dict,target_channels)
    print(f"成功读取 {len(signals_dict)} 个EDF文件")

    # II. 预处理
    # 滤波处理
    filtered_data = {}
    for file_name, signals in signals_dict.items():
        filtered_signals = []
        for ch in range(signals.shape[0]):
            filtered_ch = butter_bandpass_filter(
                signals[ch], lowcut, highcut, Fs
            )
            filtered_signals.append(filtered_ch)
        filtered_data[file_name] = np.array(filtered_signals)

    # 标准化处理
    normalized_data = {}
    for file_name, signals in filtered_data.items():
        mean = np.mean(signals, axis=1, keepdims=True)
        std = np.std(signals, axis=1, keepdims=True)
        normalized = (signals - mean) / std
        normalized_data[file_name] = normalized

    # III. 数据分段
    # 获取癫痫发作时间
    seizure_events = get_files_seizures(files_dict)

    # 划分发作期/间期/前期
    preictal_dict, interictal_dict, ictal_dict = makeEEGClips(
        Fs,  # 使用降采样后的频率
        normalized_data,
        seizure_events,
        ictal, interictal, preictal
    )

    # 数据分段
    preictal_clip = slice_data(preictal_dict, slice_samples)
    interictal_clip = slice_data(interictal_dict, slice_samples)

    # IV. 数据集构建
    # 合并间期和前期数据
    patient_data = make_patientdata(preictal_clip, interictal_clip)

    return patient_data

if __name__ == "__main__":
    main_process()