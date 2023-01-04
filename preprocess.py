
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from helper_code import *
from wave_plot import *

# Define murmur and outcome classes.
murmur_classes = ['Present', 'Unknown', 'Absent']
outcome_classes = ['Abnormal', 'Normal']


# return wave, sr
def load_wave(data_dir, wav_file):
    return torchaudio.load(os.path.join(data_dir, wav_file))


# return wave, sr
#    Using segment labels in tsv_file, cut the beginning and end of the wave
#    so that it contains whole cycles of segments 1-2-3-4.
#    Because the tsv_file was created not perfectly (badly), there might
#    be bad cycles like 4-3-1-4. But I think that were due bad segmentation.
#
#    Often, wave starts and ends with segment labeled 0. We will cut these parts.
#    It's possible the segments left may still be labeled 0. But I believe
#    it is a mistake of the segmentation program.
def get_wave_whole_cycles(data_dir, wav_file, tsv_file):
    wave, sr = load_wave(data_dir, wav_file)

    # start = rows[i][1], i is the first one so that
    #   rows[i][2] = 0 or 4 and rows[i+1][2] = 1
    #
    # end = rows[j][1], j is the last one so that
    #   rows[j][2] = 4

    with open(os.path.join(data_dir, tsv_file)) as file:
        reader = csv.reader(file, delimiter='\t')

        last_seg = '0'
        start = 0.0
        end = 0.0
        found_start = False
        for row in reader:
            if not found_start:
                if row[2] == '1' and (last_seg == '0' or last_seg == '4'):
                    start = float(row[0])
                    found_start = True
                else:
                    last_seg = row[2]
            else:
                if row[2] == '4':
                    end = float(row[1])
        wav = wave[0][int(start * sr): int(end * sr)]
        return {'wav': wav, 'seg': None}


# return  { 'wav': wav, 'seg': seg }
#    segment 1:  wav[seg[0], seg[1]]
#    segment 2:  wav[seg[1], seg[2]]
#    ...
#
# The S1 wave is identified by the integer 1.
# The systolic period is identified by the integer 2.
# The S2 wave is identified by the integer 3.
# The diastolic period is identified by the integer 4.
# The unannotated segments of the signal are identified by the integer 0.
def get_wave(data_dir, wav_file, tsv_file, segment=False, except_on_error=False):
    wav, sr = torchaudio.load(os.path.join(data_dir, wav_file))
    if not segment:
        return {'wav': wav[0], 'seg': None}
    with open(os.path.join(data_dir, tsv_file)) as file:
        rows = csv.reader(file, delimiter='\t')
        segs = list()
        next_time = 0.0
        next_seg = 0
        wave_start = 0
        for i, row in enumerate(rows):
            if next_seg == 0:  # have not found segment 1 yet
                if int(row[2]) == 0:
                    next_time = float(row[1])
                    wave_start = int(next_time * sr)
                    continue
                else:
                    next_seg = 1
            if float(row[0]) != next_time:
                raise Exception(f"segment file {tsv_file} has period gap start at {next_time}")
            if int(row[2]) == 0:
                if i == rows.line_num - 1:
                    segs.append(int(next_time * sr))
                    break
                elif except_on_error:
                    raise Exception(f"segment file {tsv_file} has period gap start at {next_time}")
                else:
                    row[2] = str(next_seg)
            if int(row[2]) != next_seg:
                if except_on_error:
                    raise Exception(f"segment file {tsv_file} has period gap start at {next_time}")
            segs.append(int(next_time * sr))
            next_time = float(row[1])
            next_seg += 1
            if next_seg == 5:
                next_seg = 1

    wave = wav[0][wave_start: int(next_time*sr)]
    if wave.size(dim=0) == 0:
        raise Exception(f"got zero length wave")
    return {'wav': wave, 'seg': segs}


"""
 
    Given  data_dir, get_data(data_dir) returns a list of patient data; data for a patient
    is a dict of keys: 'id', 'waves', 'age', 'sex', 'height', 'weight', 'pregnancy'

    where 'waves' is a dict = { 'PV': PV_wave, 'TV': TV_wave, ... }

        PV corresponds to the pulmonary valve point;
        TV corresponds to the tricuspid valve point;
        AV corresponds to the aortic valve point;
        MV corresponds to the mitral valve point;
        Phc corresponds to any other auscultation Location. (ignored)

    each wave = { 'wav': wav, 'seg': seg, 'murmur': bool }
        segment 1 wave is the slice  wav[seg[0], seg[1]]
        segment 2 wave is the slice  wav[seg[1], seg[2]]
        ...
        'murmur' indicates if murmur is present in this wave
        (The slice in the wave file that is labeled 0 is not in wav[])

"""


def get_data(data_dir, verbose):
    patient_files = find_patient_files(data_dir)  # all .txt files
    num_patient_files = len(patient_files)
    data_list = list()
    for i in range(num_patient_files):
        if verbose > 2:
            print('    {}/{}...'.format(i + 1, num_patient_files))
        data = load_patient_data(patient_files[i])
        current_locations = get_locations(data)
        num_current_locations = len(current_locations)
        recording_information = data.split('\n')[1:num_current_locations + 1]
        murmur_locations = get_murmur_locations(data)

        waves = {}
        num_errors = 0
        patient_murmur = get_murmur(data)
        for j in range(num_current_locations):
            entries = recording_information[j].split(' ')
            loc = entries[0]
            try:
                wave = get_wave_whole_cycles(data_dir, entries[2], entries[3])
            except:
                num_errors += 1
                continue
            murmur = False
            if murmur_locations is not None:
                murmur = loc in murmur_locations
            wave['murmur'] = murmur
            waves[loc] = wave
            murmur_str = patient_murmur
            if murmur:
                murmur_str = 'Present'
            # wave_plot = wave['wav']
            # wave_plot = wave_plot.clip(-0.25, 0.25)
            # if murmur_str != 'Absent':
            # plot_one_sr_waveform(np.expand_dims(wave_plot, 0), 4000, title=murmur_str)

        d = {'id': get_patient_id(data), 'waves': waves, 'age': get_age(data), 'sex': get_sex(data),
             'height': get_height(data), 'weight': get_weight(data),
             'pregnancy': get_pregnancy_status(data), 'murmur': patient_murmur, 'outcome': get_outcome(data)}
        data_list.append(d)
        if num_errors > 0:
            print(f"WARN {num_errors} were found ")
    return data_list

# return ids, mfccs, murmurs, outcomes
# each murmur ['Present', 'Unknown', 'Absent']
# each outcome ['Abnormal', 'Normal']


def pad_wave(wave, length, padding):
    wave_len = wave.size(dim=0)
    multiple = length // wave_len + 1
    if padding == 'repeat':
        wave = wave.expand((multiple, wave_len)).flatten()
    elif padding == 'zero_left':
        wave = F.pad(wave, (0, length - wave_len), "constant", 0)
    elif padding == 'zero_center':
        left = (length - wave_len) // 2
        wave = F.pad(wave, (left, length - left), "constant", 0)
    return wave[: length]


def get_ids_waves_murmurs_outcomes(data_dir, config, verbose):
    """
    returns ids, waves, murmurs, outcomes

    ids: (942, 2) 942 patients of (id, num_waves_so_far)
    waves: (num_waves, 4000*seconds), seconds = config['seconds_per_wave']
    murmurs: (num_waves, 3)
    outcomes: (num_waves, 2)
    """
    data = get_data(data_dir, verbose)
    num_patients = len(data)
    ids = np.zeros((num_patients, 2), dtype=int)
    seconds_per_wave = config.getint('seconds_per_wave')
    wave_stride = config.getint('wave_stride')
    print(f"seconds_per_wave={seconds_per_wave}  wave_stride={wave_stride}")
    wave_len = 4000 * seconds_per_wave
    stride = 4000 * wave_stride
    # win_length = config.getint('win_length')
    # hop_length = config.getint('hop_length')
    # wave_len += win_length - hop_length  # this extra size is treated as padding
    torch.manual_seed(0)   # fix seed so that test dataset remain in test even after rerun
    rand_indices = torch.randperm(num_patients)

    # find num_waves = 14391
    # pat_num_waves = list()
    num_waves = 0
    for i in range(num_patients):
        patient_data = data[rand_indices[i]]
        n_waves_for_this_patient = 0
        for location_wave in patient_data['waves'].values():
            n = location_wave['wav'].size(dim=0)
            if n < wave_len:
                location_wave['wav'] = F.pad(location_wave['wav'], (0, wave_len - n), "constant", 0)
                n_waves_for_this_patient += 1
            else:
                n_waves_for_this_patient += (n - wave_len) // stride + 1
        num_waves += n_waves_for_this_patient
        ids[i, 0] = patient_data['id']
        ids[i, 1] = num_waves
        # pat_num_waves.append(n_waves)
    # print(f"total num waves = {np.sum(pat_num_waves)}")
    waves = torch.zeros((num_waves, wave_len), dtype=torch.float32)
    murmurs = torch.zeros((num_waves, len(murmur_classes)), dtype=torch.float32)
    outcomes = torch.zeros((num_waves, len(outcome_classes)), dtype=torch.float32)

    num_waves = 0
    for i in range(num_patients):
        patient_data = data[rand_indices[i]]
        location_waves = patient_data['waves'].values()
        for location_wave in location_waves:
            j_murmur = murmur_classes.index('Absent')
            if location_wave['murmur']:
                j_murmur = murmur_classes.index('Present')
            elif compare_strings(patient_data['murmur'], 'Unknown'):
                j_murmur = murmur_classes.index('Unknown')
            j_outcome = outcome_classes.index(patient_data['outcome'])

            wave = location_wave['wav']
            num_strides = (wave.size(dim=0) - wave_len) // stride + 1
            start = 0
            for k in range(num_strides):
                waves[num_waves] = wave[start : start + wave_len]

                if verbose > 2:
                    print(f"setting patient {i} len {wave.size(dim=0)} stride {k} wave {num_waves}")
                start += stride
                murmurs[num_waves, j_murmur] = 1
                outcomes[num_waves, j_outcome] = 1
                num_waves += 1

    return ids, waves, murmurs, outcomes



def get_ids_mfccs_murmurs_outcomes(data_dir, padding, n_fft, hop_length, n_mels, verbose):
    data = get_data(data_dir, verbose)

    # find max_len of waves
    len_list = list()
    for patient_data in data:
        for location_wave in patient_data['waves'].values():
            len_list.append(location_wave['wav'].size(dim=0))
    # print_stat('wave lengths', len_list)
    num_waves = len(len_list)
    max_len = np.max(len_list)

    SAMPLE_RATE = 4000  # we have checked all waves have the same SAMPLE_RATE
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    time_banks = max_len // hop_length + 1
    ids = np.zeros((len(data), 2), dtype=int)
    mfccs = torch.zeros((num_waves, n_mels, time_banks), dtype=torch.float32)
    murmurs = torch.zeros((num_waves, len(murmur_classes)), dtype=torch.float32)
    outcomes = torch.zeros((num_waves, len(outcome_classes)), dtype=torch.float32)
    num_data = len(data)
    rand_indices = torch.randperm(num_data)
    next_patient_wave_idx = 0
    i = 0
    for patient_idx in range(len(data)):
        patient_data = data[rand_indices[patient_idx]]
        ids[patient_idx, 0] = patient_data['id']
        location_waves = patient_data['waves'].values()
        next_patient_wave_idx += len(location_waves)
        ids[patient_idx, 1] = next_patient_wave_idx
        for location_wave in location_waves:
            mfcc = mel_spectrogram(pad_wave(location_wave['wav'], max_len, padding))
            for j in range(n_mels):
                mfccs[i, j] = mfcc[j]
            j = murmur_classes.index('Absent')
            if location_wave['murmur']:
                j = murmur_classes.index('Present')
            elif compare_strings(patient_data['murmur'], 'Unknown'):
                j = murmur_classes.index('Unknown')
            murmurs[i, j] = 1
            j = outcome_classes.index(patient_data['outcome'])
            outcomes[i, j] = 1
            i += 1

    return ids, mfccs, murmurs, outcomes

# transform each wave into MFCC to form X, zero padded; Y is murmur = [ 'Present', 'Absent'] (no 'Unknown')


def get_mfcc_murmur2(data_dir, n_fft, hop_length, n_mels, verbose):
    data = get_data(data_dir, verbose)
    # find max_len of waves
    max_len = 0
    num_waves = 0
    len_list = list()
    for patient_data in data:
        for location_wave in patient_data['waves'].values():
            num_waves += 1
            wave = location_wave['wav']
            n = wave.size(dim=0)
            len_list.append(n)
            if n == 0:
                raise
            if n > max_len:
                max_len = n
    # print_stat('wave lengths', len_list)
    SAMPLE_RATE = 4000
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    time_banks = max_len // hop_length + 1
    signals = torch.zeros((num_waves, n_mels, time_banks), dtype=torch.float32)
    murmurs = torch.zeros((num_waves, 2), dtype=torch.float32)
    i = 0
    len_list = list()
    for patient_data in data:
        for location_wave in patient_data['waves'].values():
            wav = location_wave['wav']
            wav_len = wav.size(dim=0)
            repeat = max_len // wav_len + 1
            wav1 = wav.expand((repeat, wav_len))
            wav2 = wav1.flatten()
            wav3 = wav2[: max_len]
            mfcc = mel_spectrogram(wav3)
            n = mfcc.size(dim=1)
            len_list.append(n)
            for j in range(n_mels):
                # if n < time_banks:
                #     for k in range(n):
                #         signals[i, j, k] = mfcc[j, k]
                # elif n == time_banks:
                signals[i, j] = mfcc[j]
            if location_wave['murmur']:
                murmurs[i, 0] = 1  # 'Present'
            else:
                murmurs[i, 1] = 1  # 'Absent'
            i += 1
    # print_stat('time banks', len_list)
    return signals, murmurs


# transform each wave into MFCC to form X, zero padded; Y is murmur = [ 'Present', 'Unknown', 'Absent']
def get_mfcc3(data_dir, n_fft, hop_length, n_mels, verbose):
    data = get_data(data_dir, verbose)
    # data = data[800:]
    # find max_len of waves
    max_len = 0
    num_waves = 0
    len_list = list()
    for patient_data in data:
        for location_wave in patient_data['waves'].values():
            num_waves += 1
            wave = location_wave['wav']
            n = wave.size(dim=0)
            len_list.append(n)
            if n == 0:
                raise
            if n > max_len:
                max_len = n
    # print_stat('wave lengths', len_list)
    SAMPLE_RATE = 4000
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    max_len = 255600
    time_banks = max_len // hop_length + 1
    signals = torch.zeros((num_waves, n_mels, time_banks), dtype=torch.float32)
    murmurs = torch.zeros((num_waves, 3), dtype=torch.float32)
    i = 0
    len_list = list()
    for patient_data in data:
        for location_wave in patient_data['waves'].values():
            wav = location_wave['wav']
            wav_len = wav.size(dim=0)
            repeat = max_len // wav_len + 1
            wav1 = wav.expand((repeat, wav_len))
            wav2 = wav1.flatten()
            wav3 = wav2[: max_len]
            mfcc = mel_spectrogram(wav3)
            n = mfcc.size(dim=1)
            len_list.append(n)
            for j in range(n_mels):
                signals[i, j] = mfcc[j]
            if location_wave['murmur']:
                murmurs[i, 0] = 1  # 'Present'
            elif compare_strings(patient_data['murmur'], 'Unknown'):
                murmurs[i, 1] = 1
            else:
                murmurs[i, 2] = 1  # 'Absent'
            i += 1
    # print_stat('time banks', len_list)
    return signals, murmurs


# transform each wave into MFCC to form X, zero padded; Y is outcome = [ 'Normal', 'Abnormal']
def get_mfcc_outcome(data_dir, n_fft, hop_length, n_mels, verbose):
    data = get_data(data_dir, verbose)
    # find max_len of waves
    max_len = 0
    num_waves = 0
    len_list = list()
    for patient_data in data:
        for location_wave in patient_data['waves'].values():
            num_waves += 1
            wave = location_wave['wav']
            n = wave.size(dim=0)
            len_list.append(n)
            if n == 0:
                raise
            if n > max_len:
                max_len = n
    # print_stat('wave lengths', len_list)
    SAMPLE_RATE = 4000
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    time_banks = max_len // hop_length + 1
    signals = torch.zeros((num_waves, n_mels, time_banks), dtype=torch.float32)
    outcomes = torch.zeros((num_waves, 2), dtype=torch.float32)
    i = 0
    len_list = list()
    for patient_data in data:
        for location_wave in patient_data['waves'].values():
            wav = location_wave['wav']
            wav_len = wav.size(dim=0)
            repeat = max_len // wav_len + 1
            wav1 = wav.expand((repeat, wav_len))
            wav2 = wav1.flatten()
            wav3 = wav2[: max_len]
            mfcc = mel_spectrogram(wav3)
            n = mfcc.size(dim=1)
            len_list.append(n)
            for j in range(n_mels):
                signals[i, j] = mfcc[j]
            if compare_strings(patient_data['outcome'], 'Normal'):
                outcomes[i, 0] = 1
            else:
                outcomes[i, 1] = 1  # 'Abnormal'
            i += 1
    # print_stat('time banks', len_list)
    return signals, outcomes

class SimpleDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.x)


def load_data(config):
    SAMPLE_RATE = 4000
    data_dir = config.get('data_dir')
    verbose = config.getint('verbose')

    # n_mels more than 3 is useless, since some freq_bins will be empty
    transform = None
    do_transformation = config.getboolean('mel_spectrogram')
    n_mfcc = config.getint('n_mffc')
    n_fft = config.getint('n_fft')
    win_length = config.getint('win_length')
    hop_length = config.getint('hop_length')
    n_mels = config.getint('n_mels')
    power = config.getint('power', 1)
    padding = config.get('padding')
    print(f"mel_spectrogram={do_transformation} n_mffc={n_mfcc} n_fft={n_fft} win_length={win_length}")
    print(f"hop_length={hop_length}  n_mels={n_mels} power={power}  padding={padding}")
    if do_transformation:
        if n_mfcc > 0:
            transform = torchaudio.transforms.MFCC(
                sample_rate=SAMPLE_RATE,
                n_mfcc=n_mfcc,
                log_mels=True,
                melkwargs=dict(
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    power=power  # energy, 2 for power
                ))
        else:
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                power=power
            )

    ids, waves, murmurs, outcomes = get_ids_waves_murmurs_outcomes(data_dir, config, verbose)
    num_patients = len(ids)
    if verbose > 2:
        print(f"ids = {ids}")
        for wave in waves:
            plot_one_sr_waveform(np.expand_dims(wave, 0), SAMPLE_RATE)
            if do_transformation:
                mfcc = transform(wave)
                plot_spectrogram(mfcc, SAMPLE_RATE)
                for i in range(n_mfcc):
                    print_stat(f"mfcc[{i}]", mfcc[i].numpy())

    waves.unsqueeze_(dim=1)  # (14391, 1, 16000)
    # create 4 datasets/dataloaders: (train, test) * (murmur, outcome)
    batch_size = config.getint('batch_size')
    training_percent = config.getint('training_percent')
    print(f"training_percent={training_percent}%  batch_size={batch_size}")

    num_train_patients = num_patients * training_percent // 100
    num_train_waves = ids[num_train_patients - 1, 1]
    print(f"num_train_patients={num_train_patients}  num_train_waves={num_train_waves}")

    test_ids = ids[num_train_patients:]
    test_ids[:, 1] -= num_train_waves

    train_murmur_dataset = SimpleDataset(waves[:num_train_waves], murmurs[:num_train_waves], transform)
    train_outcome_dataset = SimpleDataset(waves[:num_train_waves], outcomes[:num_train_waves], transform)
    train_murmur_dataloader = DataLoader(train_murmur_dataset, batch_size)
    train_outcome_dataloader = DataLoader(train_outcome_dataset, batch_size)
    test_murmur_dataset = SimpleDataset(waves[num_train_waves:], murmurs[num_train_waves:], transform)
    test_outcome_dataset = SimpleDataset(waves[num_train_waves:], outcomes[num_train_waves:], transform)
    return train_murmur_dataloader, train_outcome_dataloader, test_murmur_dataset, test_outcome_dataset, test_ids


if __name__ == "__main__":
    data_dir = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3\\training_data'
    verbose = 1
    data = get_data(data_dir, verbose)
    print(f"got {len(data)} data")
    print(f"data[0] = {data[0]}")

    # plot murmur waves
    if verbose > 1:
        for patient_data in data:
            for location_wave in patient_data['waves'].values():
                wav = location_wave['wav']
                wav_len = wav.shape[0]
                # wav_diff = np.diff(wav)
                n = 4000
                X = np.linspace(0, n, n)
                if location_wave['murmur']:
                    plt.plot(X, wav[:n])
                    # plt.plot(X, wav_diff[:n])
                    plt.show()

    n_fft = 128
    hop_length = 60   # max_len = 152080, try to get 300 time banks, hop_length = 507 or 508
    n_mels = 4
    signals, murmurs = get_mfcc_outcome(data_dir, n_fft, hop_length, n_mels, verbose)
    print(f"got {signals.size(dim=0)} mfcc")
    print(f"signal size = {signals.size()}")
    for i in range(10):
        plot_spectrogram(signals[i])
        plot_mel_fbank(signals[i])
    # print(f"mfcc[0] = {signals[0]}")
    count = torch.count_nonzero(murmurs, dim=0)
    print(f"count murmurs {count}")
    print(f"murmurs[0] = {murmurs[0]}")
