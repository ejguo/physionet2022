import configparser
import os
import sys

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms

from models import Basic_model

from preprocess import get_ids_waves_murmurs_outcomes, SimpleDataset


class EvalDataset(Dataset):
    def __init__(self, x, transform=None):
        self.x = x
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.x)


# to be rewritten
def get_eval_data(config):
    verbose = config.getint('verbose')
    SAMPLE_RATE = 4000
    wave_file = config.get('wave_file')
    wave, sr = torchaudio.load(wave_file)
    resample = transforms.Resample(sr, SAMPLE_RATE)
    wave = resample(wave)
    seconds_per_wave = config.getint('seconds_per_wave')
    wave_len = SAMPLE_RATE * seconds_per_wave
    wave = wave[:, 0:wave_len]

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

    # num_patients = len(ids)
    num_patients = 1

    waves = wave.unsqueeze_(dim=0)  # (1, 637952) => (1, 1, 637952)

    dataset = EvalDataset(waves, transform)
    return dataset


def predict_basic(output_dir, name, config, dataset, device, verbose=0):
    model = Basic_model(config).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, f"{name}_model")))
    num_samples = len(dataset)
    y_size = config.getint('num_classes')
    label_array = torch.zeros((num_samples, y_size), dtype=torch.float32)
    prob_array = torch.zeros((num_samples, y_size), dtype=torch.float32)
    with torch.set_grad_enabled(False):
        model.eval()
        for i in range(len(dataset)):
            x = dataset[i]
            x = x.to(device)
            y1 = model(x.unsqueeze(0))[0]
            prob_array[i] = y1
            label_array[i][torch.argmax(y1)] = 1

        return label_array, prob_array


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = configparser.ConfigParser()
    config_path = sys.argv[1]
    config.read(config_path)
    output_dir = os.path.dirname(config_path)
    data_debug = config['DEFAULT'].get('data_dir')
    verbose = config['DEFAULT'].getint('verbose')
    if verbose > 0:
        print(f"Using {device}")

    eval_dataset = get_eval_data(config['preprocess'])

    murmur_labels, murmur_probs = predict_basic(output_dir, 'murmur_basic', config['murmur_basic'], eval_dataset,  device, verbose)
    print(murmur_probs)
    print(murmur_labels)
    
    outcome_labels, outcome_probs = predict_basic(output_dir, 'outcome_basic', config['outcome_basic'], eval_dataset, device, verbose)
    print(outcome_probs)
    print(outcome_labels)