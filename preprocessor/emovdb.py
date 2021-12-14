import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import csv
import pandas as pd
import codecs

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

    data, sentences = load_emov_db(in_dir)

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


    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        speaker, emotion, text, wav_path = row['speaker'], row['emotion'], row['text'], row['path']
        base_name = os.path.split(wav_path)[-1].split(".")[0]
        text = _clean_text(text, cleaners)
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sampling_rate)
            # wav, index = librosa.effects.trim(wav, frame_length=filter_length, hop_length=hop_length, top_db=30)
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
                            text = sentences[sentences['n'] == fnumber]['text'].iloc[
                                0]  # result must be a string and not a df with a single element
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