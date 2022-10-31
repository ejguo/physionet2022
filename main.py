import os
import sys

import numpy as np
import torch
import configparser

from evaluation_model import print_evaluation, print_patient_evaluation
from models import build_model
from preprocess import load_data
from train_helper import train, test


# This replaces old file 'train_and_eval.py' after forming the functions
#  load_data()   in preprocess.py
#  build_model() in models.py
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    train_config = config['train']
    verbose = train_config.getint('verbose')
    if verbose > 0:
        print(f"Using {device}")

    train_murmur_dataloader, train_outcome_dataloader, test_murmur_dataset, test_outcome_dataset, test_ids = load_data(config['preprocess'])
    murmur_model, outcome_model, murmur_loss_fn, outcome_loss_fn, murmur_optimiser, outcome_optimiser = build_model(train_config, device)

    epochs = train_config.getint('epochs')
    output_dir = train_config.get('output_dir')
    train(murmur_model, train_murmur_dataloader, murmur_loss_fn, murmur_optimiser, device, epochs)
    torch.save(murmur_model.state_dict(), os.path.join(output_dir, f"murmur_model"))
    murmur_labels, murmur_probs = test(murmur_model, test_murmur_dataset, murmur_loss_fn, device, verbose)

    train(outcome_model, train_outcome_dataloader, outcome_loss_fn, outcome_optimiser, device, epochs)
    torch.save(outcome_model.state_dict(), os.path.join(output_dir, f"outcome_model"))
    outcome_labels, outcome_probs = test(outcome_model, test_outcome_dataset, outcome_loss_fn, device, verbose)

    print_evaluation(murmur_labels, murmur_probs, outcome_labels, outcome_probs)

    data_dir = train_config.get('data_dir')
    print_patient_evaluation(data_dir, test_ids, murmur_probs, outcome_probs, verbose)
