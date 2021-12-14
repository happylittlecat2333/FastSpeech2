import re
import os
import json
from string import punctuation
from text import text_to_sequence
from g2p_en import G2p
from pypinyin import pinyin, Style

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import librosa
from scipy.io import wavfile
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import audio as Audio

matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_variance_level(preprocess_config, model_config, data_loading=True):
    """
    Consider the fact that there is no pre-extracted phoneme-level variance features in unsupervised duration modeling.
    Outputs:
        pitch_level_tag, energy_level_tag: ["frame", "phone"]
            If data_loading is set True, then it will only be the "frame" for unsupervised duration modeling. 
            Otherwise, it will be aligned with the feature_level in config.
        pitch_feature_level, energy_feature_level: ["frame_level", "phoneme_level"]
            The feature_level in config where the model will learn each variance in this level regardless of the input level.
    """
    # learn_alignment = model_config["duration_modeling"]["learn_alignment"] if data_loading else False
    pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
    energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
    assert pitch_feature_level in ["frame_level", "phoneme_level"]
    assert energy_feature_level in ["frame_level", "phoneme_level"]
    pitch_level_tag = "frame" if pitch_feature_level == "frame_level" else "phone"
    energy_level_tag = "frame" if energy_feature_level == "frame_level" else "phone"
    # pitch_level_tag = "phone" if (not learn_alignment and pitch_feature_level == "phoneme_level") else "frame"
    # energy_level_tag = "phone" if (not learn_alignment and energy_feature_level == "phoneme_level") else "frame"
    return pitch_level_tag, energy_level_tag, pitch_feature_level, energy_feature_level


def get_phoneme_level_pitch(duration, pitch):
    # perform linear interpolation
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))

    # Phoneme-level average
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            pitch[i] = np.mean(pitch[pos:pos + d])
        else:
            pitch[i] = 0
        pos += d
    pitch = pitch[:len(duration)]
    return pitch


def get_phoneme_level_energy(duration, energy):
    # Phoneme-level average
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            energy[i] = np.mean(energy[pos:pos + d])
        else:
            energy[i] = 0
        pos += d
    energy = energy[:len(duration)]
    return energy


def to_device(data, device):

    if data["speakers"] is not None:
        data["speakers"] = torch.from_numpy(data["speakers"]).long().to(device)
    if data["emotions"] is not None:
        data["emotions"] = torch.from_numpy(data["emotions"]).long().to(device)
    if data["texts"] is not None:
        data["texts"] = torch.from_numpy(data["texts"]).long().to(device)
    if data["src_lens"] is not None:
        data["src_lens"] = torch.from_numpy(data["src_lens"]).to(device)
    if data["mels"] is not None:
        data["mels"] = torch.from_numpy(data["mels"]).float().to(device)
    if data["mel_lens"] is not None:
        data["mel_lens"] = torch.from_numpy(data["mel_lens"]).to(device)
    if data["pitches"] is not None:
        data["pitches"] = torch.from_numpy(data["pitches"]).float().to(device)
    if data["energies"] is not None:
        data["energies"] = torch.from_numpy(data["energies"]).float().to(device)
    if data["durations"] is not None:
        data["durations"] = torch.from_numpy(data["durations"]).long().to(device)
    if data["spker_embeds"] is not None:
        data["spker_embeds"] = torch.from_numpy(data["spker_embeds"]).float().to(device)

    return data


def loss_dict(loss_list):
    loss_dict = {}
    loss_name = [
        "Total Loss",
        "Mel Loss",
        "Mel PostNet Loss",
        "Pitch Loss",
        "Energy Loss",
        "Duration Loss",
        "Speaker_loss_1",
        "Speaker_loss_2",
        "Emotion_loss_1",
        "Emotion_loss_2",
        "Emotion_loss_1_revgrad",
        "Emotion_loss_2_revgrad",
        "Speaker_Emotion_loss_1",
        "Speaker_Emotion_loss_2",
        "Emotion_Style_loss",
        "Loss_1",
        "Loss_2",
        "All_loss",

    ]
    assert len(loss_list) == len(loss_name)
    for idx in range(len(loss_list)):
        loss = loss_list[idx]
        if loss is None or loss == 0.:
            continue
        loss_dict[loss_name[idx]] = loss
    return loss_dict


def get_audio(preprocess_config, wav_path):

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    STFT = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        preprocess_config["preprocessing"]["stft"]["hop_length"],
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )

    # Read and trim wav files
    wav, _ = librosa.load(wav_path, sampling_rate).astype(np.float32)

    # Compute mel-scale spectrogram
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, STFT)
    mels = mel_spectrogram.T.astype(np.float32)

    return mels


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_raw_text(text, preprocess_config, lang="en"):
    if lang == "en":
        text = text.rstrip(punctuation)
        lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

        g2p = G2p()
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in lexicon:
                phones += lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", g2p(w)))
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")

    elif lang == "zh":
        lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
        phones = []
        pinyins = [p[0] for p in pinyin(text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)]
        for p in pinyins:
            if p in lexicon:
                phones += lexicon[p]
            else:
                phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]))

    return np.array(sequence)



def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        for k, v in losses.items():
            if v:
                logger.add_scalar("Loss/{}".format(k), v, step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config, idx):

    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)

    basename = targets["ids"][idx]
    src_len = predictions["src_lens"][idx].item()
    mel_len = predictions["mel_lens"][idx].item()
    mel_target = targets["mels"][idx, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions["postnet_output"][idx, :mel_len].detach().transpose(0, 1)
    duration = predictions["d_preds"][idx, :src_len].detach().cpu().numpy()

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets["pitches"][idx, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets["pitches"][idx, :mel_len].detach().cpu().numpy()

    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets["energies"][idx, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets["energies"][idx, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats[f"pitch_{pitch_level_tag}"] + stats[f"energy_{energy_level_tag}"][:2] # Should follow the level at data loading time.

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path, args):

    basenames = targets["ids"]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions["src_lens"][i].item()
        mel_len = predictions["mel_lens"][i].item()
        mel_prediction = predictions["postnet_output"][i, :mel_len].detach().transpose(0, 1)
        duration = predictions["d_preds"][i, :src_len].detach().cpu().numpy()

        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions["p_preds"][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions["p_preds"][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions["e_preds"][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions["e_preds"][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, str(args.restore_step), "{}_{}_{}.png".format(
            basename, args.speaker_id, args.emotion_id)
            if args.mode == "single" else "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions["postnet_output"].transpose(1, 2)
    lengths = predictions["mel_lens"] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, str(args.restore_step), "{}_{}_{}.wav".format(
            basename, args.speaker_id, args.emotion_id)
            if args.mode == "single" else "{}.wav".format(basename)),
            sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def plot_mel_(fig, axes, data, stats, titles, tight_layout=True):
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean
    if tight_layout:
        fig.tight_layout()

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato", linewidth=.7)
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(labelsize="x-small", colors="tomato", bottom=False, labelbottom=False)

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet", linewidth=.7)
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )


# def plot_single_alignment(alignment, info=None, save_dir=None):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
#     fig.colorbar(im, ax=ax)
#     xlabel = 'Decoder timestep'
#     if info is not None:
#         xlabel += '\n\n' + info
#     plt.xlabel(xlabel)
#     plt.ylabel('Encoder timestep')
#     plt.tight_layout()

#     fig.canvas.draw()
#     data = save_figure_to_numpy(fig)
#     if save_dir is not None:
#         plt.savefig(save_dir)
#     plt.close()
#     return data


def plot_alignment(data, titles=None, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, figsize=[6, 4], dpi=300)
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.05)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        im = data[i]
        axes[i].imshow(im, origin='lower')
        axes[i].set_xlabel('Audio timestep')
        axes[i].set_ylabel('Text timestep')
        axes[i].set_ylim(0, im.shape[0])
        axes[i].set_xlim(0, im.shape[1])
        axes[i].set_title(titles[i], fontsize='medium')
        axes[i].tick_params(labelsize='x-small')
        axes[i].set_anchor('W')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()
    return data


def plot_embedding(out_dir, spker_embed_dict, filename='embedding.png'):

    Y, X = [], []
    for spk, emb in spker_embed_dict.items():
        for idx in range(len(emb)):
            Y.append(spk)
            X.append(emb[idx])

    tsne_model = TSNE(n_components=2, random_state=0, init="pca", n_jobs=-1)
    X_tsne = tsne_model.fit_transform(X)
    X = np.array(X_tsne)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X_norm = (X - x_min) / (x_max - x_min)

    df = pd.DataFrame(dict(Feature_1=X_norm[:, 0], Feature_2=X_norm[:, 1], label=Y))
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="Feature_1", y="Feature_2",
        hue="label",
        palette=sns.color_palette("hls", len(spker_embed_dict)),
        data=df,
        legend="full",
    )
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()




def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return data


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_3D(inputs, B, T, L):
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, :np.shape(input_)[0], :np.shape(input_)[1]] = input_
    return inputs_padded


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
