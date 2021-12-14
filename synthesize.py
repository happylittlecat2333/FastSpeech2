import os
import json
import argparse

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader


from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, get_audio, preprocess_raw_text
from dataset import TextDataset
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def synthesize(model, step, configs, vocoder, batchs, control_values, args):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                batch,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                args,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore_step",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--emotion_id",
        type=int,
        default=0,
        help="emotion ID for multi-emotion synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="reference audio path to extract the speech style, for single-sentence mode only",
    )

    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml"
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config, model_config, args)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        batchs = defaultdict(lambda: None)
        batchs["ids"] = batchs["raw_texts"] = [args.text[:100]]

        # Speaker, Emotion Map
        load_spker_embed = model_config["multi_speaker"] and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
            emotions_map = json.load(f)


        # Speaker, Emotion, Speaker Embedding
        batchs["speakers"] = np.array([speaker_map[args.speaker_id]])
        batchs["spker_embeds"] = np.load(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"],
                "spker_embed",
                preprocess_config["preprocessing"]["speaker_embedder"],
                "{}-spker_embed.npy".format(args.speaker_id),
            )) if load_spker_embed else None

        # either use ref_audio or emotion_id
        assert args.ref_audio ^ args.emotions
        if args.ref_audio:
            batchs["mels"] = get_audio(preprocess_config, args.ref_audio)
            batchs["mel_lens"] = np.array([batchs["mels"].shape[0]])
            batchs["max_mel_lens"] = max(batchs["mel_lens"])
        else:
            batchs["emotions"] = np.array([emotions_map[args.emotion_id]])

        lang = preprocess_config["preprocessing"]["text"]["language"]
        batchs["texts"] = np.array([preprocess_raw_text(args.text, preprocess_config, lang)])

        batchs["src_lens"] = np.array([len(batchs["texts"][0])])
        batchs["max_src_len"] = max(batchs["src_lens"])

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)

"""
BATCH SYNTHESIS:

CUDA_VISIBLE_DEVICES=0 python synthesize.py \
    --restore_step 900000 \
    --mode batch \
    --source val.txt \
    -p  exp/EmovDB/test/config/preprocess.yaml \
    -m  exp/EmovDB/test/config/model.yaml \
    -t  exp/EmovDB/test/config/train.yaml \

SINGLE SYNTHESIS:

CUDA_VISIBLE_DEVICES=0 python synthesize.py \
    --restore_step 900000 \
    --mode single \
    --text "I'm so happy to meet you" \
    --speaker_id 0 \
    --emotion_id 0  \ or --ref_audio /path/to/audio.wav \
    --pitch_control 1.0 \
    --energy_control 1.0 \
    --duration_control 1.0 \
    -p  exp/EmovDB/test/config/preprocess.yaml \
    -m  exp/EmovDB/test/config/model.yaml \
    -t  exp/EmovDB/test/config/train.yaml \

"""