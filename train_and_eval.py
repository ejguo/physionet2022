"""

    data -> mfcc, murmurs, outcomes -> Dataset(mfcc, murmurs), Dataset(mfcc, outcomes)

    model, dataloader, loss_fn, optimiser, epochs, model_name -> train and save to "{model_name}.model"

    "{model_name}.model" -> model

    test_dataloader, model -> predictions

    murmurs, murmurs_predictions, outcomes, outcomes_predictions -> eval

"""
import os
import sys
import torchvision.transforms as transforms
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from evaluation_model import print_evaluation, print_patient_evaluation
from models import C2F3, MurmurModel1, WaveCNN, C2F2, FcNN, C4F1_1, C4F1_raw, C4F1, C4F1_2
from preprocess import get_ids_mfccs_murmurs_outcomes, murmur_classes, outcome_classes, get_ids_waves_murmurs_outcomes
from train_helper import train, SimpleDataset, test, check_dataloader
import configparser

from wave_plot import plot_one_sr_waveform, plot_spectrogram, print_stat

if __name__ == "__main__":
    SAMPLE_RATE = 4000
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    preprocess_config = config['preprocess']
    data_dir = preprocess_config.get('data_dir')
    output_dir = preprocess_config.get('output_dir')
    verbose = preprocess_config.getint('verbose')
    print(f"config: {sys.argv[1]}  data_dir={data_dir}  output_dir={output_dir} verbose={verbose}")

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose > 0:
        print(f"Using {device}")

    # n_mels more than 3 is useless, since some freq_bins will be empty
    transform = None
    do_transformation = preprocess_config.getboolean('mel_spectrogram')
    n_mfcc = preprocess_config.getint('n_mffc')
    n_fft = preprocess_config.getint('n_fft')
    win_length = preprocess_config.getint('win_length')
    hop_length = preprocess_config.getint('hop_length')
    n_mels = preprocess_config.getint('n_mels')
    power = preprocess_config.getint('power', 1)
    padding = preprocess_config.get('padding')
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

    # ids, mfccs, murmurs, outcomes = get_ids_mfccs_murmurs_outcomes(data_dir, padding, n_fft, hop_length, n_mels, verbose)
    ids, waves, murmurs, outcomes = get_ids_waves_murmurs_outcomes(data_dir, preprocess_config, verbose)
    # waves_1 = torch.diff(waves, dim=1)
    # waves = torch.cat((waves[:,1:],waves_1), dim=1)
    # waves = nn.functional.normalize(waves, dim=1)
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
    train_config = config['train']
    batch_size = train_config.getint('batch_size')
    training_percent = train_config.getint('training_percent')
    print(f"training_percent={training_percent}%  batch_size={batch_size}")

    num_train_patients = num_patients * training_percent // 100
    num_train_waves = ids[num_train_patients - 1, 1]
    print(f"num_train_patients={num_train_patients}  num_train_waves={num_train_waves}")

    train_murmur_dataset = SimpleDataset(waves[:num_train_waves], murmurs[:num_train_waves], transform)
    train_outcome_dataset = SimpleDataset(waves[:num_train_waves], outcomes[:num_train_waves], transform)
    train_murmur_dataloader = DataLoader(train_murmur_dataset, batch_size)
    train_outcome_dataloader = DataLoader(train_outcome_dataset, batch_size)
    # test_murmur_dataset = SimpleDataset(waves[num_train_waves:], murmurs[num_train_waves:], transform)
    test_murmur_dataset = SimpleDataset(waves[:num_train_waves], murmurs[:num_train_waves], transform)  # test
    # test_outcome_dataset = SimpleDataset(waves[num_train_waves:], outcomes[num_train_waves:], transform)
    test_outcome_dataset = SimpleDataset(waves[:num_train_waves], outcomes[:num_train_waves], transform)  # test

    num_murmurs = len(murmur_classes)
    num_outcomes = len(outcome_classes)
    epochs = train_config.getint('epochs')
    learning_rate = train_config.getfloat('learning_rate')
    dropout_rate = train_config.getfloat('dropout_rate')
    model_type = train_config.get('model_type')
    print(f"epochs={epochs}  learning_rate={learning_rate}")
    murmur_model = None
    outcome_model = None
    if model_type == 'MurmurModel1':
        murmur_model = MurmurModel1(n_mels, num_murmurs, dropout_rate).to(device)
        outcome_model = MurmurModel1(n_mels, num_outcomes, dropout_rate).to(device)
    elif model_type == 'C2F3':
        murmur_model = C2F3(n_mels, num_murmurs).to(device)
        outcome_model = C2F3(n_mels, num_outcomes).to(device)
    elif model_type == 'WaveCNN':
        murmur_model = WaveCNN(1, num_murmurs).to(device)
        outcome_model = WaveCNN(1, num_outcomes).to(device)
    elif model_type == 'C2F2':
        murmur_model = C2F2(n_mfcc, num_murmurs).to(device)
        outcome_model = C2F2(n_mfcc, num_outcomes).to(device)
    elif model_type == 'FcNN':
        wave_len = 4000 * preprocess_config.getint('wave_seconds')
        murmur_model = FcNN(wave_len, num_murmurs).to(device)
        outcome_model = FcNN(wave_len, num_outcomes).to(device)
    elif model_type == 'C4F1':
        murmur_model = C4F1(num_murmurs, dropout_rate).to(device)
        outcome_model = C4F1(num_outcomes, dropout_rate).to(device)
    elif model_type == 'C4F1_raw':
        murmur_model = C4F1_raw(num_murmurs).to(device)
        outcome_model = C4F1_raw(num_outcomes).to(device)
    elif model_type == 'C4F1_1':
        murmur_model = C4F1_1(num_murmurs, dropout_rate).to(device)
        outcome_model = C4F1_1(num_outcomes, dropout_rate).to(device)
    elif model_type == 'C4F1_2':
        murmur_model = C4F1_2(num_murmurs, dropout_rate).to(device)
        outcome_model = C4F1_2(num_outcomes, dropout_rate).to(device)

    print(f"Murmur model: {murmur_model}")

    loss_murmur_weights = np.array(train_config.get('loss_murmur_weights').split(','), dtype=float)
    print(f"loss_murmur_weights={loss_murmur_weights} CrossEntropyLoss Adam")
    murmur_loss_fn = nn.CrossEntropyLoss(torch.tensor(loss_murmur_weights)).to(device)
    murmur_optimiser = torch.optim.Adam(murmur_model.parameters(), lr=learning_rate)
    train(murmur_model, train_murmur_dataloader, murmur_loss_fn, murmur_optimiser, device, epochs)
    torch.save(murmur_model.state_dict(), os.path.join(output_dir, f"murmur_{model_type}"))

    murmur_labels, murmur_probs = test(murmur_model, test_murmur_dataset, murmur_loss_fn, device, verbose)

    print(f"Outcome model: {outcome_model}")
    loss_outcome_weights = np.array(train_config.get('loss_outcome_weights').split(','), dtype=float)
    print(f"loss_outcome_weights={loss_outcome_weights}  CrossEntropyLoss Adam")
    outcome_loss_fn = nn.BCELoss(torch.tensor(loss_outcome_weights)).to(device)
    outcome_optimiser = torch.optim.Adam(outcome_model.parameters(), lr=learning_rate)
    if verbose > 2:
        check_dataloader(train_murmur_dataloader, ids[:num_train_patients])
    train(outcome_model, train_outcome_dataloader, outcome_loss_fn, outcome_optimiser, device, epochs)
    torch.save(outcome_model.state_dict(), os.path.join(output_dir, f"outcome_{model_type}"))

    outcome_labels, outcome_probs = test(outcome_model, test_outcome_dataset, outcome_loss_fn, device, verbose)

    print_evaluation(murmur_labels, murmur_probs, outcome_labels, outcome_probs)

    test_ids = ids[num_train_patients:]
    test_ids[:, 1] -= num_train_waves
    print_patient_evaluation(data_dir, test_ids, murmur_probs, outcome_probs, verbose)
