# %%

import os
import pandas as pd
import codecs
import logging
import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

import sys
sys.path.append('../')

from text import _clean_text
# from ..text import _clean_text
# %%

def load_emov_db(path_to_EmoV_DB):
    transcript = os.path.join(path_to_EmoV_DB, 'cmuarctic.data')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()

    # in our database, we use only files beginning with arctic_a. And the number of these sentences correspond.
    # Here we build a dataframe with number and text of each of these lines
    sentences = []
    for line in lines:
        temp = {}
        idx_n_0 = line.find('arctic_a') + len('arctic_a')
        if line.find('arctic_a') != -1:
            print(line)
            print(idx_n_0)
            idx_n_end = idx_n_0 + 4
            number = line[idx_n_0:idx_n_end]
            print(number)
            temp['n'] = number
            idx_text_0 = idx_n_end + 2
            text = line.strip()[idx_text_0:-3]
            temp['text'] = text
            # print(text)
            sentences.append(temp)
    sentences = pd.DataFrame(sentences)

    print(sentences)
    speakers = next(os.walk(path_to_EmoV_DB))[1]  #this list directories (and not files, contrary to osl.listdir() )

    data = []

    for spk in speakers:
        print(spk)
        emo_cat = next(os.walk(os.path.join(
            path_to_EmoV_DB, spk)))[1]  #this list directories (and not files, contrary to osl.listdir() )

        for emo in emo_cat:
            for file in os.listdir(os.path.join(path_to_EmoV_DB, spk, emo)):
                print(file)
                fpath = os.path.join(path_to_EmoV_DB, spk, emo, file)

                if file[-4:] == '.wav':
                    fnumber = file[-8:-4]
                    print(fnumber)
                    if fnumber.isdigit():
                        try:
                            text = sentences[sentences['n'] == fnumber]['text'].iloc[0]  # result must be a string and not a df with a single element
                        except:
                            continue
                        # text_lengths.append(len(text))
                        # texts.append(text)
                        # texts.append(np.array(text, np.int32).tostring())
                        # fpaths.append(fpath)
                        # emo_cats.append(emo)

                        e = {
                            'database': 'EmoV-DB',
                            'id': file[:-4],
                            'speaker': spk,
                            'emotion': emo,
                            'transcription': text,
                            'sentence_path': fpath
                        }
                        data.append(e)
                        print(e)

    data = pd.DataFrame.from_records(data)

    return data, sentences


path = '/home/samba/public/Datasets/emotion/EmoV-DB'


# %%

data, sentences = load_emov_db(path)


# %%
import csv

speakers = data['speaker'].values
emotions = data['emotion'].apply(lambda x: x[4:] if x[:2] == "au" else x)
emotions.replace(to_replace=['am', 'neut', 'sleep'],
                 value=['amused', 'neutral', 'sleepiness'],
                 inplace=True)
emotions = emotions.values
texts = data['transcription']
paths = data['sentence_path']
ids = data['id'].values

df = pd.DataFrame({"speaker": speakers, "emotion": emotions, "text": texts, "path": paths, "id": ids})
df.to_csv("test.csv", index=False)
df.to_csv("/home/samba/public/Datasets/emotion/EmoV-DB/emovdb.csv", index=False)


emotion_dict = {
    "neutral": "01",
    "amused": "02",
    "disgust": "03",
    "sleepiness": "04",
    "anger": "05",
    "calm": "06",
    "fearful": "07",
    "surprised": "08",
}

def parse_filename(spk, emo, path):
    return "_".join([spk, emo, path.split("/")[-1].split(".")[0]])



def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    for index, row in df.iterrows():
        speaker, emo, text, wav_path = row['speaker'], row['emotion'], row['text'], row['path']
        base_name = parse_filename(speaker, emo, wav_path)
        text = _clean_text(text, cleaners)
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
            ) as f1:
                f1.write(text)
        else:
            print("[Error] No flac file:{}".format(wav_path))

import yaml

args_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/EmovDB/preprocess.yaml"

config = config = yaml.load(open(args_path, "r"), Loader=yaml.FullLoader)
prepare_align(config)

# %%

import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

intensity_dict = {
    "01": "normal",
    "02": "strong",
}

script_dict = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


def parse_filename(filename, speaker):
    res = []
    name_list = filename.split("-")
    res.append(emotion_dict[name_list[2]])
    res.append(intensity_dict[name_list[3]])
    res.append(name_list[5])
    res.append(speaker)
    return "-".join(res), script_dict[name_list[4]]


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    for spker_id, speaker in enumerate(tqdm(os.listdir(in_dir))):
        if "Actor_" not in speaker:
            continue
        for i, wav_name in enumerate(tqdm(os.listdir(os.path.join(in_dir, speaker)))):
            wav_path = os.path.join(os.path.join(in_dir, speaker, wav_name))
            base_name, text = parse_filename(wav_name.split(".")[0], speaker)
            text = _clean_text(text, cleaners)

            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                        "w",
                ) as f1:
                    f1.write(text)
            else:
                print("[Error] No flac file:{}".format(wav_path))
                continue
