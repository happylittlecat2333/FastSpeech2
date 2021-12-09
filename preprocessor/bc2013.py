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
    speaker = "bc2013"
    emotion = "neutral"

    with open(os.path.join(in_dir, 'prompts.gui')) as prompts_file:
        lines = [l[:-1] for l in prompts_file]

    wav_paths = [
        os.path.join(in_dir, 'wavn', fname + '.wav')
        for fname in lines[::3]
    ]

    base_names = [fname for fname in lines[::3]]

    # Clean up the transcripts
    transcripts = lines[1::3]
    for i in range(len(transcripts)):
        t = transcripts[i]
        t = t.replace('@ ', '')
        t = t.replace('# ', '')
        t = t.replace('| ', '')
        t = t.lower()
        transcripts[i] = t


    total = len(wav_paths)

    for base_name, text, wav_path in tqdm(zip(base_names, transcripts, wav_paths), total=total):
        text = _clean_text(text, cleaners)
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