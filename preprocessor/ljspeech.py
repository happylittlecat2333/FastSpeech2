import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    filter_length = config["preprocessing"]["stft"]["filter_length"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    speaker = "LJSpeech"
    emotion = "neutral"

    file_path = os.path.join(in_dir, "metadata.csv")
    total = len(open(file_path).readlines(  ))

    with open(file_path, encoding="utf-8") as f:
        for line in tqdm(f, total=total):
            parts = line.strip().split("|")
            base_name = parts[0]
            text = parts[2]
            text = _clean_text(text, cleaners)

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav, index = librosa.effects.trim(wav, frame_length=filter_length, hop_length=hop_length, top_db=60)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}_{}_{}.wav".format(speaker, emotion, base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}_{}_{}.lab".format(speaker, emotion, base_name)),
                    "w",
                ) as f1:
                    f1.write(text)