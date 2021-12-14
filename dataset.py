from collections import defaultdict
import json
import math
import os
import argparse

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import get_variance_level, pad_1D, pad_2D, pad_3D
from utils.tools import preprocess_raw_text, get_audio


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

        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]
        self.mel_normalization = preprocess_config["preprocessing"]["mel"]["normalization"]

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
            "normalize" if self.mel_normalization else "raw",
            "{}-{}-mel-{}.npy".format(speaker, emotion, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch_{}".format(self.pitch_level_tag),
            "normalize" if self.pitch_normalization else "raw",
            "{}-{}-pitch-{}.npy".format(speaker, emotion, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy_{}".format(self.energy_level_tag),
            "normalize" if self.energy_normalization else "raw",
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
        # spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
        #     if self.load_spker_embed else None
        spker_embeds = np.array([data[idx]["spker_embed"] for idx in idxs]) \
            if self.load_spker_embed else None

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        ids = np.array(ids)
        raw_texts = np.array(raw_texts)
        speakers = np.array(speakers)
        emotions = np.array(emotions)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)


        return defaultdict(lambda: None, {
            "ids": ids,
            "raw_texts": raw_texts,
            "speakers": speakers,
            "emotions": emotions,
            "texts": texts,
            "src_lens": text_lens,
            "max_src_len": max(text_lens),
            "mels": mels,
            "mel_lens": mel_lens,
            "max_mel_len": max(mel_lens),
            "pitches": pitches,
            "energies": energies,
            "durations": durations,
            "spker_embeds": spker_embeds,
        })

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
    def __init__(self, filepath, preprocess_config, model_config, args):
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.ref_audio = args.ref_audio

        self.lang = preprocess_config["preprocessing"]["text"]["language"]

        self.basename, self.speaker, self.emotion, self.text, self.raw_text = self.process_meta(filepath)

        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]
        self.mel_normalization = preprocess_config["preprocessing"]["mel"]["normalization"]

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
        if not self.text[idx]:
            phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        else:
            phone = np.array([preprocess_raw_text(raw_text, preprocess_config, self.lang)])
        if self.ref_audio:
            if basename.endswith(".wav"):
                mel = get_audio(self.preprocess_config, basename)
            else:
                mel_path = os.path.join(
                    self.preprocessed_path,
                    "mel",
                    "normalize" if self.mel_normalization else "raw",
                    "{}-{}-mel-{}.npy".format(speaker, emotion, basename),
                )
                mel = np.load(mel_path)
        else:
            mel = None
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            self.embedder_type,
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None


        return {
            "basename": basename,
            "speaker_id": speaker_id,
            "emotion_id": emotion_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "spker_embed": spker_embed,
        }

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
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

    def collate_fn(self, data):
        ids = [d["basename"] for d in data]
        speakers = np.array([d["speaker_id"] for d in data])
        emotions = np.array([d["emotion_id"] for d in data])
        texts = [d["text"] for d in data]
        raw_texts = [d["raw_text"] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        mels = [d["mel"] for d in data] if self.ref_audio else None
        mel_lens = np.array([mel.shape[0] for mel in mels]) if self.ref_audio else None
        max_mel_lens = max(mel_lens) if self.ref_audio else None
        # spker_embeds = np.concatenate(np.array([d["spker_embed"] for d in data]), axis=0) \
        #     if self.load_spker_embed else None
        spker_embeds = np.array([d["spker_embed"] for d in data]) \
            if self.load_spker_embed else None

        texts = pad_1D(texts)

        return defaultdict(lambda: None, {
            "ids": ids,
            "raw_texts": raw_texts,
            "speakers": speakers,
            "emotions": emotions,
            "texts": texts,
            "src_lens": text_lens,
            "max_src_len": max(text_lens),
            "mels": mels,
            "mel_lens": mel_lens,
            "max_mel_len": max_mel_lens,
            "spker_embeds": spker_embeds,
        })


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
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
    # parser.add_argument(
    #     "-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml"
    # )
    # parser.add_argument(
    #     "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    # )
    # parser.add_argument(
    #     "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    # )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(open("./config/EmovDB/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("./config/EmovDB/train.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("./config/EmovDB/model.yaml", "r"), Loader=yaml.FullLoader)

    train_dataset = Dataset("train.txt", preprocess_config, model_config, train_config, sort=True, drop_last=True)
    val_dataset = Dataset("val.txt", preprocess_config, model_config, train_config, sort=False, drop_last=False)
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
            print(batch)
            for k,v in batch.items():
                print(k, type(v))

            out = to_device(batch, device)
            n_batch += 1
    print("Training set  with size {} is composed of {} batches.".format(len(train_dataset), n_batch))

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print("Validation set  with size {} is composed of {} batches.".format(len(val_dataset), n_batch))

    dataset = TextDataset(args.source, preprocess_config, model_config, args)
    batchs = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=dataset.collate_fn,
    )
    for batch in batchs:
        print(batch)
        for k,v in batch.items():
            print(k, type(v))
