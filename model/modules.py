import os, sys
import json
import copy
import math
from collections import OrderedDict
import numpy


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools import (
    get_variance_level,
    get_phoneme_level_pitch,
    get_phoneme_level_energy,
    get_mask_from_lengths,
    pad_1D,
    pad,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .transformers.blocks import LinearNorm, ConvNorm


# @jit(nopython=True)
# def mas_width1(attn_map):
#     """mas with hardcoded width=1"""
#     # assumes mel x text
#     opt = np.zeros_like(attn_map)
#     attn_map = np.log(attn_map)
#     attn_map[0, 1:] = -np.inf
#     log_p = np.zeros_like(attn_map)
#     log_p[0, :] = attn_map[0, :]
#     prev_ind = np.zeros_like(attn_map, dtype=np.int64)
#     for i in range(1, attn_map.shape[0]):
#         for j in range(attn_map.shape[1]):  # for each text dim
#             prev_log = log_p[i - 1, j]
#             prev_j = j

#             if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
#                 prev_log = log_p[i - 1, j - 1]
#                 prev_j = j - 1

#             log_p[i, j] = attn_map[i, j] + prev_log
#             prev_ind[i, j] = prev_j

#     # now backtrack
#     curr_text_idx = attn_map.shape[1] - 1
#     for i in range(attn_map.shape[0] - 1, -1, -1):
#         opt[i, curr_text_idx] = 1
#         curr_text_idx = prev_ind[i, curr_text_idx]
#     opt[0, curr_text_idx] = 1
#     return opt


# @jit(nopython=True, parallel=True)
# def b_mas(b_attn_map, in_lens, out_lens, width=1):
#     assert width == 1
#     attn_out = np.zeros_like(b_attn_map)

#     for b in prange(b_attn_map.shape[0]):
#         out = mas_width1(b_attn_map[b, 0, :out_lens[b], :in_lens[b]])
#         attn_out[b, 0, :out_lens[b], :in_lens[b]] = out
#     return attn_out


class SpeakerEmbedding(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(SpeakerEmbedding, self).__init__()
        assert model_config["multi_speaker"] == True
        self.embedder_type =  model_config["speaker_embedding"]["embedder_type"]
        self.encoder_type = model_config["block_type"]
        self.encoder_dim = model_config[self.encoder_type]["encoder_hidden"]
        if self.embedder_type == "embedding":
            with open(
                    os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"),
                    "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                self.encoder_dim,
            )
        else:
            self.speaker_emb = nn.Linear(
                model_config["speaker_embedding"]["pretrained_model"][self.embedder_type]["speaker_dim"],
                self.encoder_dim,
            )

    def forward(self, speakers, spker_embeds):
        if self.embedder_type == "embedding":
            speaker_emb = self.speaker_emb(speakers)
        else:
            speaker_emb = self.speaker_emb(spker_embeds)
        return speaker_emb


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            ))

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                ))

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            ))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""
    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)


        pitch_level_tag, energy_level_tag, self.pitch_feature_level, self.energy_feature_level = \
                        get_variance_level(preprocess_config, model_config, data_loading=False)

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]

        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats[f"pitch_{pitch_level_tag}"][:2]
            energy_min, energy_max = stats[f"energy_{energy_level_tag}"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])
        self.energy_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(torch.bucketize(prediction, self.energy_bins))
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            min=0,
        )
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, p_control)
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:

            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mel_mask, p_control)
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """
    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict([
                (
                    "conv1d_1",
                    ConvNorm(
                        self.input_size,
                        self.filter_size,
                        kernel_size=self.kernel,
                        stride=1,
                        padding=(self.kernel - 1) // 2,
                        dilation=1,
                        transpose=True,
                    ),
                ),
                ("relu_1", nn.ReLU()),
                ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                ("dropout_1", nn.Dropout(self.dropout)),
                (
                    "conv1d_2",
                    ConvNorm(
                        self.filter_size,
                        self.filter_size,
                        kernel_size=self.kernel,
                        stride=1,
                        padding=1,
                        dilation=1,
                        transpose=True,
                    ),
                ),
                ("relu_2", nn.ReLU()),
                ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                ("dropout_2", nn.Dropout(self.dropout)),
            ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class GlobalEmotionToken(nn.Module):
    """ Global Emotion Token """
    def __init__(self, preprocess_config, model_config):
        super(GlobalEmotionToken, self).__init__()
        self.encoder = ReferenceEncoder(preprocess_config, model_config)
        self.etl = ETL(preprocess_config, model_config)

    def forward(self, inputs, emotions):
        enc_out = None
        if inputs is not None:
            if not self.training:
                assert emotions is None
            enc_out = self.encoder(inputs)
        else:
            if not self.training:
                assert emotions is not None
        emotion_embed_hard, emotion_embed_soft, score_hard, score_soft = self.etl(enc_out, emotions)

        return emotion_embed_hard, emotion_embed_soft, score_hard, score_soft


class ReferenceEncoder(nn.Module):
    """ Reference Mel Encoder """
    def __init__(self, preprocess_config, model_config):
        super(ReferenceEncoder, self).__init__()

        E = model_config["transformer"]["encoder_hidden"]
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        ref_enc_filters = model_config["emotion_token_layer"]["ref_enc_filters"]
        ref_enc_size = model_config["emotion_token_layer"]["ref_enc_size"]
        ref_enc_strides = model_config["emotion_token_layer"]["ref_enc_strides"]
        ref_enc_pad = model_config["emotion_token_layer"]["ref_enc_pad"]
        ref_enc_gru_size = model_config["emotion_token_layer"]["ref_enc_gru_size"]

        self.n_mel_channels = n_mel_channels
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            nn.Conv2d(in_channels=filters[i],
                      out_channels=filters[i + 1],
                      kernel_size=ref_enc_size,
                      stride=ref_enc_strides,
                      padding=ref_enc_pad) for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels, hidden_size=E // 2, batch_first=True)

    def forward(self, inputs):
        """
        inputs --- [N, Ty/r, n_mels*r]
        outputs --- [N, ref_enc_gru_size]
        """
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.n_mel_channels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class ETL(nn.Module):
    """ Emotion Token Layer """
    def __init__(self, preprocess_config, model_config):
        super(ETL, self).__init__()

        E = model_config["transformer"]["encoder_hidden"]
        num_heads = model_config["emotion_token_layer"]["num_heads"]
        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json"),
                "r",
        ) as f:
            token_num = len(json.load(f))

        self.token_num = token_num
        self.embed = nn.Parameter(torch.FloatTensor(token_num, E // num_heads))
        d_q = E // 2
        d_k = E // num_heads
        self.attention = StyleEmbedAttention(query_dim=d_q, key_dim=d_k, num_units=E, num_heads=num_heads)

        nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs, score_hard=None):
        if inputs is not None:
            N = inputs.size(0)
            query = inputs.unsqueeze(1)  # [N, 1, E//2]
        else:
            N = score_hard.size(0)
            query = None
        keys_soft = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        if score_hard is not None:
            score_hard = F.one_hot(score_hard, self.token_num).float().detach()  # [N, token_num]
        emotion_embed_hard, emotion_embed_soft, score_soft = self.attention(query, keys_soft, score_hard)

        return emotion_embed_hard, emotion_embed_soft, score_hard, score_soft.squeeze(0).squeeze(
            1) if score_soft is not None else None


class StyleEmbedAttention(nn.Module):
    """ StyleEmbedAttention """
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super(StyleEmbedAttention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key_soft, score_hard=None):
        """
        input:
            query --- [N, T_q, query_dim]
            key_soft --- [N, T_k, key_dim]
            score_hard --- [N, T_k]
        output:
            out --- [N, T_q, num_units]
        """
        values = self.W_value(key_soft)
        split_size = self.num_units // self.num_heads
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        out_hard = out_soft = scores_soft = None
        if query is not None:
            querys = self.W_query(query)  # [N, T_q, num_units]
            keys = self.W_key(key_soft)  # [N, T_k, num_units]

            # [h, N, T_q, num_units/h]
            querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
            # [h, N, T_k, num_units/h]
            keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
            # [h, N, T_k, num_units/h]

            # score = softmax(QK^T / (d_k ** 0.5))
            scores_soft = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
            scores_soft = scores_soft / (self.key_dim**0.5)
            scores_soft = F.softmax(scores_soft, dim=3)

            # out = score * V
            # [h, N, T_q, num_units/h]
            out_soft = torch.matmul(scores_soft, values)
            out_soft = torch.cat(torch.split(out_soft, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        if score_hard is not None:
            # [N, T_k] -> [h, N, T_q, T_k]
            score_hard = score_hard.unsqueeze(0).unsqueeze(2).repeat(self.num_heads, 1, 1, 1)
            out_hard = torch.matmul(score_hard, values)
            out_hard = torch.cat(torch.split(out_hard, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out_hard, out_soft, scores_soft




class StyleEncoder(torch.nn.Module):
    """Style encoder.
    This module is style encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.
    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.
    Todo:
        * Support manual weight specification in inference.
    """
    def __init__(
        self,
        idim: int = 80,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: list = [32, 32, 64, 64, 128, 128],
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        super(StyleEncoder, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )
        self.stl = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
        Returns:
            Tensor: Style token embeddings (B, token_dim).
        """
        ref_embs = self.ref_enc(speech)
        style_embs = self.stl(ref_embs)

        return style_embs


class ReferenceEncoder(torch.nn.Module):
    """Reference encoder module.
    This module is reference encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.
    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.
    """
    def __init__(
        self,
        idim=80,
        conv_layers: int = 6,
        conv_chans_list: list = [32, 32, 64, 64, 128, 128],
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize reference encoder module."""
        super(ReferenceEncoder, self).__init__()

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (len(conv_chans_list) == conv_layers
                ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False,
                ),
                torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
            ]
        self.convs = torch.nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (gru_in_units - conv_kernel_size + 2 * padding) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, idim).
        Returns:
            Tensor: Reference embedding (B, gru_units)
        """
        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (B, 1, Lmax, idim)
        hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
        # NOTE(kan-bayashi): We need to care the length?
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)  # (B, Lmax', gru_units)
        self.gru.flatten_parameters()
        _, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        ref_embs = ref_embs[-1]  # (batch_size, gru_units)

        return ref_embs


class StyleTokenLayer(torch.nn.Module):
    """Style token layer module.
    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.
    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.
    """
    def __init__(
        self,
        ref_embed_dim: int = 128,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initilize style token layer module."""
        super(StyleTokenLayer, self).__init__()

        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter("gst_embs", torch.nn.Parameter(gst_embs))
        self.mha = MultiHeadedAttention(
            q_dim=ref_embed_dim,
            k_dim=gst_token_dim // gst_heads,
            v_dim=gst_token_dim // gst_heads,
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, ref_embs: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            ref_embs (Tensor): Reference embeddings (B, ref_embed_dim).
        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).
        """
        batch_size = ref_embs.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        # NOTE(kan-bayashi): Shoule we apply Tanh?
        ref_embs = ref_embs.unsqueeze(1)  # (batch_size, 1 ,ref_embed_dim)
        style_embs = self.mha(ref_embs, gst_embs, gst_embs, None)

        return style_embs.squeeze(1)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        """Initialize multi head attention module."""
        # NOTE(kan-bayashi): Do not use super().__init__() here since we want to
        #   overwrite BaseMultiHeadedAttention.__init__() method.
        torch.nn.Module.__init__(self)
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(q_dim, n_feat)
        self.linear_k = torch.nn.Linear(k_dim, n_feat)
        self.linear_v = torch.nn.Linear(v_dim, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k))  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class VAE(nn.Module):
    """
    Variational AutoEncoder
    """
    def __init__(self, model_config):
        super(VAE, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.hidden_size = model_config["vae"]["hidden_size"]
        self.latent_size = model_config["vae"]["latent_size"]

        self.mu = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar = nn.Linear(self.hidden_size, self.latent_size)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
        else:
            z = mu

        return z, mu, logvar


class SpeakerClassifier(nn.Module):
    """
    Speaker Classifier
    """
    def __init__(self, preprocess_config, model_config):
        super(SpeakerClassifier, self).__init__()

        self.hidden_size = model_config["speaker_classifier"]["hidden_size"]
        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"),
                "r",
        ) as f:
            self.n_speaker = len(json.load(f))
        self.fc = nn.Linear(self.hidden_size, self.n_speaker)

    def forward(self, x):
        x = self.fc(x)
        return x


class EmotionClassifier(nn.Module):
    """
    Emotion Classifier
    """
    def __init__(self, preprocess_config, model_config):
        super(EmotionClassifier, self).__init__()

        self.hidden_size = model_config["emotion_classifier"]["hidden_size"]
        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json"),
                "r",
        ) as f:
            self.n_emotion = len(json.load(f))
        self.fc = nn.Linear(self.hidden_size, self.n_emotion)

    def forward(self, x):
        x = self.fc(x)
        return x


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)




class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, input):
        batch, dim = input.shape
        input = input.unsqueeze(-1)
        return torch.bmm(input, input.transpose(1, 2)) / (2.0 * batch * dim)

    def forward(self, input, target):
        assert len(input.shape) == 2
        input = self.gram_matrix(input)
        target = self.gram_matrix(target)
        return F.mse_loss(input, target)


class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    def forward(self, input, target):
        assert len(input.shape) == 2
        batch, dim = input.shape
        input = input.unsqueeze(1)
        target = target.unsqueeze(1)
        # print(dim)
        ort_loss = torch.bmm(input, target.transpose(1, 2)) / dim
        return F.mse_loss(ort_loss, torch.zeros_like(ort_loss, requires_grad=False))


def test():
    import yaml
    model_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LibriTTS_emovDB/model.yaml"
    train_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LibriTTS_emovDB/train.yaml"
    preprocess_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LibriTTS_emovDB/preprocess.yaml"

    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)

    model = VAE(model_config)
    model.train()
    x = torch.randn(16, 256)
    y = model(x)
    z, mu, logvar = y
    print(z.shape, mu.shape, logvar.shape)

    model.eval()
    y = model(x)


def test2():
    x = torch.randn(16, 256)
    y = torch.randn(16, 256)
    loss = StyleLoss()
    print(loss.gram_matrix(x).shape)
    print(loss(x, y))

if __name__ == "__main__":
    # test()
    test2()