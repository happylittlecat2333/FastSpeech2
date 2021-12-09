import os
import re
import random
import json
import copy

import tgt
import librosa
import numpy as np
import torch
import pyworld as pw
from scipy.stats import betabinom
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

import audio as Audio
from utils.tools import get_phoneme_level_pitch, get_phoneme_level_energy, plot_embedding
from model.speaker_embedding import PreDefinedEmbedder

emotion_dict = {
    "neutral": 0,
    "amused": 1,
    "disgust": 2,
    "sleepiness": 3,
    "anger": 4,
    "calm": 5,
    "fearful": 6,
    "surprised": 7,
}


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config['seed'])

        self.preprocess_config = preprocess_config
        self.multi_speaker = model_config['multi_speaker']
        self.in_dir = preprocess_config["path"]["raw_path"]
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

        assert preprocess_config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]
        self.mel_normalization = preprocess_config["preprocessing"]["mel"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )

        self.speaker_emb = None
        self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
        self.in_sub_dirs = [p for p in os.listdir(self.in_dir) if os.path.isdir(os.path.join(self.in_dir, p))]
        if self.multi_speaker and preprocess_config["preprocessing"]["speaker_embedder"] != "none":
            self.speaker_emb = PreDefinedEmbedder(preprocess_config, model_config)
            self.speaker_emb_dict = self._init_spker_embeds(self.in_sub_dirs)

    def _init_spker_embeds(self, spkers):
        spker_embeds = dict()
        for spker in spkers:
            spker_embeds[spker] = list()
        return spker_embeds

    def build_from_path(self):
        embedding_dir = os.path.join(self.out_dir, "spker_embed", self.embedder_type)
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((embedding_dir), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        max_seq_len = -float('inf')

        pitch_frame_scaler = StandardScaler()
        pitch_phone_scaler = StandardScaler()
        energy_frame_scaler = StandardScaler()
        energy_phone_scaler = StandardScaler()
        mel_scaler = StandardScaler()

        def partial_fit(scaler, value):
            if len(value) > 0:
                if len(value.shape) == 1:
                    scaler.partial_fit(value.reshape(-1, 1))
                else:
                    scaler.partial_fit(value)

        def compute_stats(scaler, dir, normalization=True):
            if normalization:
                mean_ = scaler.mean_
                std_ = scaler.scale_
            else:
                mean_ = 0
                std_ = 1
            min_, max_ = self.normalize(os.path.join(self.out_dir, dir), mean_, std_)
            return min_, max_, mean_, std_

        # def compute_stats(pitch_scaler, energy_scaler, pitch_dir="pitch", energy_dir="energy"):
        #     if self.pitch_normalization:
        #         pitch_mean = pitch_scaler.mean_[0]
        #         pitch_std = pitch_scaler.scale_[0]
        #     else:
        #         # A numerical trick to avoid normalization...
        #         pitch_mean = 0
        #         pitch_std = 1
        #     if self.energy_normalization:
        #         energy_mean = energy_scaler.mean_[0]
        #         energy_std = energy_scaler.scale_[0]
        #     else:
        #         energy_mean = 0
        #         energy_std = 1

        #     pitch_min, pitch_max = self.normalize(
        #         os.path.join(self.out_dir, pitch_dir), pitch_mean, pitch_std
        #     )
        #     energy_min, energy_max = self.normalize(
        #         os.path.join(self.out_dir, energy_dir), energy_mean, energy_std
        #     )
        #     return (pitch_min, pitch_max, pitch_mean, pitch_std), (energy_min, energy_max, energy_mean, energy_std)


        skip_speakers = set()
        for embedding_name in os.listdir(embedding_dir):
            skip_speakers.add(embedding_name.split("-")[0])

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers, emotions, emotion_set = {}, {}, set()
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            save_speaker_emb = self.speaker_emb is not None and speaker not in skip_speakers
            if os.path.isdir(os.path.join(self.in_dir, speaker)):
                speakers[speaker] = i
                for ii, wav_name in enumerate(tqdm(os.listdir(os.path.join(self.in_dir, speaker)))):
                    if ".wav" not in wav_name:
                        continue

                    basename = wav_name.split(".")[0]
                    emotion = basename.split("_")[1]
                    # emotion_set.add(emotion)
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                    )
                    if os.path.exists(tg_path):
                        # ret = self.process_utterance(speaker, emotion, basename)
                        ret = self.process_utterance(tg_path, speaker, emotion, basename, save_speaker_emb)
                        if ret is None:
                            continue
                        else:
                            # info, pitch_frame, pitch_phone, energy_frame, energy_phone, n = ret
                            info, pitch_frame, pitch_phone, energy_frame, energy_phone, n, mel, spker_embed = ret
                        out.append(info)

                    if save_speaker_emb:
                        self.speaker_emb_dict[speaker].append(spker_embed)

                    partial_fit(pitch_frame_scaler, pitch_frame)
                    partial_fit(pitch_phone_scaler, pitch_phone)
                    partial_fit(energy_frame_scaler, energy_frame)
                    partial_fit(energy_phone_scaler, energy_phone)
                    partial_fit(mel_scaler, mel)

                    if n > max_seq_len:
                        max_seq_len = n

                    n_frames += n

                # Calculate and save mean speaker embedding of this speaker
                if save_speaker_emb:
                    spker_embed_filename = '{}-spker_embed.npy'.format(speaker)
                    np.save(os.path.join(embedding_dir, spker_embed_filename), \
                        np.mean(self.speaker_emb_dict[speaker], axis=0), allow_pickle=False)

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        # pitch_frame_stats, energy_frame_stats = compute_stats(
        #     pitch_frame_scaler,
        #     energy_frame_scaler,
        #     pitch_dir="pitch_frame",
        #     energy_dir="energy_frame",
        # )
        # pitch_phone_stats, energy_phone_stats = compute_stats(
        #     pitch_phone_scaler,
        #     energy_phone_scaler,
        #     pitch_dir="pitch_phone",
        #     energy_dir="energy_phone",
        # )
        pitch_frame_stats = compute_stats(pitch_frame_scaler, "pitch_frame", self.pitch_normalization)
        pitch_phone_stats = compute_stats(pitch_phone_scaler, "pitch_phone", self.pitch_normalization)
        energy_frame_stats = compute_stats(energy_frame_scaler, "energy_frame", self.energy_normalization)
        energy_phone_stats = compute_stats(energy_phone_scaler, "energy_phone", self.energy_normalization)
        mel_stats = compute_stats(mel_scaler, "mel", self.mel_normalization)


        # for i, e in enumerate(emotion_set):
        #     emotions[e] = i

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
            # f.write(json.dumps(emotions))
            f.write(json.dumps(emotion_dict))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch_frame": [float(var) for var in pitch_frame_stats],
                "pitch_phone": [float(var) for var in pitch_phone_stats],
                "energy_frame": [float(var) for var in energy_frame_stats],
                "energy_phone": [float(var) for var in energy_phone_stats],
                "mel": [float(var) for var in mel_stats],
                "max_seq_len": max_seq_len
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        if self.speaker_emb is not None:
            print("Plot speaker embedding...")
            plot_embedding(
                self.out_dir, *self.load_embedding(embedding_dir),
                self.divide_speaker_by_gender(self.corpus_dir), filename="spker_embed_tsne.png"
            )

        assert len(out) == 0
        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, tg_path, speaker, emotion, basename, save_speaker_emb):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)
        # spker_embed = self.speaker_emb(wav) if save_speaker_emb else None


        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        mel_spectrogram = mel_spectrogram.T # T, C
        spker_embed = self.speaker_emb(torch.from_numpy(mel_spectrogram)).numpy() if save_speaker_emb else None


        # Frame-level variance
        pitch_frame, energy_frame = copy.deepcopy(pitch), copy.deepcopy(energy)

        # Phone-level variance
        pitch_phone, energy_phone = get_phoneme_level_pitch(duration, pitch), get_phoneme_level_energy(duration, energy)

        # Save files
        dur_filename = "{}-{}-duration-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-{}-pitch-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "pitch_frame", pitch_filename), pitch_frame)

        pitch_filename = "{}-{}-pitch-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "pitch_phone", pitch_filename), pitch_phone)

        energy_filename = "{}-{}-energy-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "energy_frame", energy_filename), energy_frame)

        energy_filename = "{}-{}-energy-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "energy_phone", energy_filename), energy_phone)

        mel_filename = "{}-{}-mel-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "mel", mel_filename), mel_spectrogram)

        return (
            "|".join([basename, speaker, emotion, text, raw_text]),
            self.remove_outlier(pitch_frame),
            self.remove_outlier(pitch_phone),
            self.remove_outlier(energy_frame),
            self.remove_outlier(energy_phone),
            mel_spectrogram.shape[1],
            mel_spectrogram,
            spker_embed,
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

    def divide_speaker_by_gender(self, in_dir, speaker_path="speaker-info.txt"):
        speakers = dict()
        with open(os.path.join(in_dir, speaker_path), encoding='utf-8') as f:
            for line in tqdm(f):
                if "ID" in line: continue
                parts = [p.strip() for p in re.sub(' +', ' ',(line.strip())).split(' ')]
                spk_id, gender = parts[0], parts[2]
                speakers[str(spk_id)] = gender
        return speakers

    def divide_speaker_by_gender(self, in_dir, speaker_path="speaker-info.txt"):
        speakers = dict()
        with open(os.path.join(in_dir, speaker_path), encoding='utf-8') as f:
            for line in tqdm(f):
                if "ID" in line: continue
                parts = [p.strip() for p in re.sub(' +', ' ',(line.strip())).split(' ')]
                spk_id, gender = parts[0], parts[2]
                speakers[str(spk_id)] = gender
        return speakers

    def load_embedding(self, embedding_dir):
        embedding_path_list = [_ for _ in Path(embedding_dir).rglob('*.npy')]
        embedding = None
        embedding_speaker_id = list()
        # Gather data
        for path in tqdm(embedding_path_list):
            embedding = np.concatenate((embedding, np.load(path)), axis=0) \
                                            if embedding is not None else np.load(path)
            embedding_speaker_id.append(str(str(path).split('/')[-1].split('-')[0]))
        return embedding, embedding_speaker_id
