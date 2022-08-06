
import csv
import os
import torch
import torchaudio

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
    raise

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

        d = {'id': get_patient_id(data), 'waves': waves, 'age': get_age(data), 'sex': get_sex(data),
             'height': get_height(data), 'weight': get_weight(data),
             'pregnancy': get_pregnancy_status(data), 'murmur': get_murmur(data), 'outcome': get_outcome(data)}
        data_list.append(d)
        if num_errors > 0:
            print(f"WARN {num_errors} were found ")
    return data_list

# return ids, mfccs, murmurs, outcomes
# each murmur ['Present', 'Unknown', 'Absent']
# each outcome ['Abnormal', 'Normal']


def get_ids_mfccs_murmurs_outcomes(data_dir, n_fft, hop_length, n_mels, verbose):
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
    next_data_idx = 0
    i = 0
    for i_data in range(len(data)):
        patient_data = data[rand_indices[i_data]]
        ids[i_data, 0] = patient_data['id']
        location_waves = patient_data['waves'].values()
        next_data_idx += len(location_waves)
        ids[i_data, 1] = next_data_idx
        for location_wave in location_waves:
            wav = location_wave['wav']
            wav_len = wav.size(dim=0)
            repeat = max_len // wav_len + 1
            wav1 = wav.expand((repeat, wav_len))
            wav2 = wav1.flatten()
            wav3 = wav2[: max_len]
            mfcc = mel_spectrogram(wav3)
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


if __name__ == "__main__":
    data_dir = 'D:\\git\\challenge2022\\the-circor-digiscope-phonocardiogram-dataset-1.0.3\\training_data'
    verbose = 1
    data = get_data(data_dir, verbose)
    print(f"got {len(data)} data")
    print(f"data[0] = {data[0]}")

    n_fft = 1024
    hop_length = 512   # max_len = 152080, try to get 300 time banks, hop_length = 507 or 508
    n_mels = 3
    signals, murmurs = get_mfcc_outcome(data_dir, n_fft, hop_length, n_mels, verbose)
    print(f"got {signals.size(dim=0)} mfcc")
    print(f"signal size = {signals.size()}")
    for i in range(10):
        plot_spectrogram(signals[i], 4000)
    # print(f"mfcc[0] = {signals[0]}")
    count = torch.count_nonzero(murmurs, dim=0)
    print(f"count murmurs {count}")
    print(f"murmurs[0] = {murmurs[0]}")
