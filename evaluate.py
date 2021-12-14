import argparse
import os
import random

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

from utils.model import get_model, get_vocoder
from utils.tools import loss_dict, to_device, log, synth_one_sample
from model.loss import FastSpeech2Loss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device).eval()

    # Evaluation
    # loss_sums = [0 for _ in range(20)]
    model.train()
    losses = defaultdict(lambda: torch.tensor(0.))
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(batch)

                # Cal Loss
                losses = Loss(batch, output)

                for k, v in losses.items():
                    losses[k] += v * len(batch["ids"])

    for k, v in losses.items():
        losses[k] /= len(dataset)


    message = ", ".join(["{}: {:.4f}".format(k, v) for k, v in losses.items() if v])
    message = "Validation Step {}, ".format(step) + message

    if logger is not None:
        for index in random.sample(range(0, len(batch)-1), 6):

            fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                index,
            )
            log(logger, step, losses=losses)
            log(
                logger,
                fig=fig,
                tag="Validation/step_{}_{}".format(step, tag),
            )
            sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
            log(
                logger,
                audio=wav_reconstruction,
                sampling_rate=sampling_rate,
                tag="Validation/step_{}_{}_reconstructed".format(step, tag),
            )
            log(
                logger,
                audio=wav_prediction,
                sampling_rate=sampling_rate,
                tag="Validation/step_{}_{}_synthesized".format(step, tag),
            )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)