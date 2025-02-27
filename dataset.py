import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import get_variance_level, pad_1D, pad_2D, pad_3D


class Dataset(Dataset):
    def __init__(self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.pitch_level_tag, self.energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)

        self.basename, self.speaker, self.emotion, self.text, self.raw_text = self.process_meta(filename)

        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        emotion = self.emotion[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[emotion]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-{}-mel-{}.npy".format(speaker, emotion, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch_{}".format(self.pitch_level_tag),
            "{}-{}-pitch-{}.npy".format(speaker, emotion, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy_{}".format(self.energy_level_tag),
            "{}-{}-energy-{}.npy".format(speaker, emotion, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-{}-duration-{}.npy".format(speaker, emotion, basename),
        )
        duration = np.load(duration_path)

        spker_embed = np.load(
            os.path.join(
                self.preprocessed_path,
                "spker_embed",
                self.embedder_type,
                "{}-spker_embed.npy".format(speaker),
            )) if self.load_spker_embed else None

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "emotion": emotion_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "spker_embed": spker_embed,

        }

        return sample

    def process_meta(self, filename):
        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:
            name = []
            speaker = []
            emotion = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, e, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                emotion.append(e)
                text.append(t)
                raw_text.append(r)
            return name, speaker, emotion, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        emotions = [data[idx]["emotion"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        emotions = np.array(emotions)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            emotions,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            spker_embeds,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[:len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.basename, self.speaker, self.emotion, self.text, self.raw_text = self.process_meta(filepath)
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
            self.emotion_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion = self.emotion[idx]
        emotion_id = self.emotion_map[emotion]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            self.embedder_type,
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None


        return (basename, speaker_id, emotion_id, phone, raw_text, spker_embed)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            emotion = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                emotion.append(t)
                text.append(t)
                raw_text.append(r)
            return name, speaker, emotion, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        emotions = np.array([d[2] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None

        texts = pad_1D(texts)

        return (
            ids,
            raw_texts,
            speakers,
            emotions,
            texts,
            text_lens,
            max(text_lens),
            spker_embeds,
        )


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(open("./config/EmovDB/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("./config/EmovDB/train.yaml", "r"), Loader=yaml.FullLoader)

    train_dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    val_dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print("Training set  with size {} is composed of {} batches.".format(len(train_dataset), n_batch))

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print("Validation set  with size {} is composed of {} batches.".format(len(val_dataset), n_batch))
