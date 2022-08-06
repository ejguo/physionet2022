"""

    data -> mfcc, murmurs, outcomes -> Dataset(mfcc, murmurs), Dataset(mfcc, outcomes)

    model, dataloader, loss_fn, optimiser, epochs, model_name -> train and save to "{model_name}.model"

    "{model_name}.model" -> model

    test_dataloader, model -> predictions

    murmurs, murmurs_predictions, outcomes, outcomes_predictions -> eval

"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation_model import print_evaluation, print_patient_evaluation
from models import SimpleCNN
from preprocess import get_ids_mfccs_murmurs_outcomes, murmur_classes, outcome_classes
from train_helper import train, SimpleDataset, test

if __name__ == "__main__":

    data_dir = 'D:\\git\\challenge2022\\the-circor-digiscope-phonocardiogram-dataset-1.0.3\\training_data'
    verbose = 2
    batch_size = 32
    epochs = 5
    learning_rate = 0.00001
    print(f"verbose={verbose}  batch_size={batch_size}  epochs={epochs}   learning_rate={learning_rate}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose > 0:
        print(f"Using {device}")

    # n_mels more than 3 is useless, since some freq_bins will be empty
    n_fft = 1024
    hop_length = 512
    n_mels = 3
    print(f"n_fft={n_fft}  hop_length={hop_length}  n_mels={n_mels}")
    ids, mfccs, murmurs, outcomes = get_ids_mfccs_murmurs_outcomes(data_dir, n_fft, hop_length, n_mels, verbose)
    if verbose > 1:
        print(f"ids = {ids}")
    num_patients = len(ids)

    # create 4 dataloaders: (train, test) * (murmur, outcome)
    training_percent = 90
    print(f"training_percent={training_percent}%")
    num_train_patients = num_patients * training_percent // 100
    num_train_waves = ids[num_train_patients - 1, 1]
    print(f"num_train_patients={num_train_patients}  num_train_waves={num_train_waves}")
    train_murmur_dataloader = DataLoader(SimpleDataset(mfccs[:num_train_waves], murmurs[:num_train_waves]), batch_size)
    train_outcome_dataloader = DataLoader(SimpleDataset(mfccs[:num_train_waves], outcomes[:num_train_waves]),
                                          batch_size)
    test_murmur_dataset = SimpleDataset(mfccs[num_train_waves:], murmurs[num_train_waves:])
    test_outcome_dataset = SimpleDataset(mfccs[num_train_waves:], outcomes[num_train_waves:])
    # train_murmur_dataloader = build_dataloader(mfccs, murmurs, train_indices, batch_size)
    # train_outcome_dataloader = build_dataloader(mfccs, outcomes, train_indices, batch_size)
    # test_murmur_dataset = build_dataset(mfccs, murmurs, test_indices)
    # test_outcome_dataset = build_dataset(mfccs, outcomes, test_indices)

    num_murmurs = len(murmur_classes)
    murmur_model = SimpleCNN(n_mels, num_murmurs).to(device)
    print(f"Murmur model SimpleCNN: {murmur_model}")
    loss_murmur_weights = [1, 2, 1]
    print(f"loss_murmur_weights={loss_murmur_weights} CrossEntropyLoss Adam")
    murmur_weights = torch.tensor(loss_murmur_weights, dtype=torch.float)
    murmur_loss_fn = nn.CrossEntropyLoss(murmur_weights).to(device)
    murmur_optimiser = torch.optim.Adam(murmur_model.parameters(), lr=learning_rate)
    train(murmur_model, train_murmur_dataloader, murmur_loss_fn, murmur_optimiser, device, epochs)
    torch.save(murmur_model.state_dict(), "murmur.model")

    murmur_labels, murmur_probs = test(murmur_model, test_murmur_dataset, murmur_loss_fn, device, verbose)

    num_outcomes = len(outcome_classes)
    outcome_model = SimpleCNN(n_mels, num_outcomes).to(device)
    print(f"Outcome model SimpleCNN: {outcome_model}")
    loss_outcome_weights = [1, 1.5]
    print(f"loss_outcome_weights={loss_outcome_weights}  CrossEntropyLoss Adam")
    outcome_weights = torch.tensor(loss_outcome_weights, dtype=torch.float)
    outcome_loss_fn = nn.CrossEntropyLoss(outcome_weights).to(device)
    outcome_optimiser = torch.optim.Adam(outcome_model.parameters(), lr=learning_rate)
    train(outcome_model, train_outcome_dataloader, outcome_loss_fn, outcome_optimiser, device, epochs)
    torch.save(outcome_model.state_dict(), "outcome.model")

    outcome_labels, outcome_probs = test(outcome_model, test_outcome_dataset, outcome_loss_fn, device, verbose)

    print_evaluation(murmur_labels, murmur_probs, outcome_labels, outcome_probs)

    test_ids = ids[num_train_patients:]
    test_ids[:, 1] -= num_train_waves
    print_patient_evaluation(data_dir, test_ids, murmur_probs, outcome_probs, verbose)
