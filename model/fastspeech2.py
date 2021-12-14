import json
import os
import sys
from typing import Text

import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import text
from utils.tools import get_mask_from_lengths
from collections import defaultdict

from model.loss import FastSpeech2Loss

from model.modules import (VAE, EmotionClassifier, GlobalEmotionToken,
                           GradientReversal, PostNet, SpeakerClassifier,
                           SpeakerEmbedding, VarianceAdaptor)





class FastSpeech2(nn.Module):
    """ FastSpeech2 """
    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        if model_config["block_type"] == "transformer":
            from model.transformers.transformer import Decoder, TextEncoder
        elif model_config["block_type"] == "lstransformer":
            from model.transformers.lstransformer import Decoder, TextEncoder
        elif model_config["block_type"] == "fastformer":
            from model.transformers.fastformer import Decoder, TextEncoder
        elif model_config["block_type"] == "conformer":
            from model.transformers.conformer import Decoder, TextEncoder
        elif model_config["block_type"] == "reformer":
            from model.transformers.reformer import Decoder, TextEncoder
        else:
            raise NotImplementedError

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )

        self.postnet = PostNet()

        if model_config["multi_speaker"]:
            if model_config["Encoder_config"]["use_speaker"]:
                self.speaker_encoder_1 = SpeakerEmbedding(preprocess_config, model_config)
                if model_config["Encoder_config"]["use_speaker_classifier"]:
                    self.speaker_classifier_1 = SpeakerClassifier(preprocess_config, model_config)

            if model_config["Decoder_config"]["use_speaker"]:
                self.speaker_encoder_2 = SpeakerEmbedding(preprocess_config, model_config)
                if model_config["Decoder_config"]["use_speaker_classifier"]:
                    self.speaker_classifier_2 = SpeakerClassifier(preprocess_config, model_config)

        if model_config["multi_emotion"]:
            if model_config["Encoder_config"]["use_emotion"]:
                self.emotion_encoder_1 = GlobalEmotionToken(preprocess_config, model_config)
                if model_config["Encoder_config"]["use_emotion_classifier"]:
                    self.emotion_classifier_1 = EmotionClassifier(preprocess_config, model_config)
                if model_config["Encoder_config"]["use_revgrad"]:
                    self.revgrad_1 = nn.Sequential(
                        GradientReversal(),
                        SpeakerClassifier(preprocess_config, model_config),
                    )

            if model_config["Decoder_config"]["use_emotion"]:
                self.emotion_encoder_2 = GlobalEmotionToken(preprocess_config, model_config)
                if model_config["Decoder_config"]["use_emotion_classifier"]:
                    self.emotion_classifier_2 = EmotionClassifier(preprocess_config, model_config)
                if model_config["Decoder_config"]["use_revgrad"]:
                    self.revgrad_2 = nn.Sequential(
                        GradientReversal(),
                        SpeakerClassifier(preprocess_config, model_config),
                    )

        self.criterion = FastSpeech2Loss(preprocess_config, model_config)

    def forward(
        self,
        inputs,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        """
        Args:
            inputs: defaultdict(lamda: None)
                - texts: (B, max_src_len)
                - mels: (B, max_mel_len, n_mel_channels)
                - src_lens, mel_lens: (B, )
                - max_src_len, max_mel_len: int
                - speakers: (B, )
                - emotions: (B, )
                - pitches, energies, durations: (B, max_mel_len)
                - speaker_embedding: (B, speaker_embedding_dim)
            p_control, e_control, d_control: float
            
        Returns: defaultdict(lamda: None)
            - output: (B, max_mel_len, n_mel_channels)
            - postnet_output: (B, max_mel_len, n_mel_channels)
            - src_lens, mel_lens: (B, )
            - src_mask: (B, max_src_len)
            - mel_mask: (B, max_mel_len)
            - p_preds, e_preds, d_preds, d_rounded: (B, max_src_len)
            - spk_emb_1, spk_emb_2: (B, speaker_embedding_dim)
            - emo_emb_1, emo_emb_2: (B, emotion_embedding_dim)
            - spk_cls_1_output, spk_cls_2_output: (B, n_speaker)
            - emo_cls_1_output, emo_cls_2_output: (B, n_emotion)
            - emo_cls_1_rev_output, emo_cls_2_rev_output: (B, n_emotion)

        """
        src_lens, max_src_len = inputs["src_lens"], inputs["max_src_len"]
        mel_lens, max_mel_len = inputs["mel_lens"], inputs["max_mel_len"]

        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None

        results = defaultdict(lambda: None)

        output, src_word_emb = self.encoder(inputs["texts"], src_masks)

        if hasattr(self, "speaker_encoder_1"):
            results["spk_emb_1"] = self.speaker_encoder_1(inputs["speakers"], inputs["spker_embeds"])
            output = output + results["spk_emb_1"].unsqueeze(1).expand(-1, max_src_len, -1)

        if hasattr(self, "emotion_encoder_1"):
            emotion_embed_hard, emotion_embed_soft, score_hard, score_soft = self.emotion_encoder_1(
                inputs["mels"], inputs["emotions"])
            # 尽量用hard模式，选择对应某个情感的token，做为输入，而不是用reference的语音得到的soft token
            results["emo_emb_1"] = emotion_embed_hard if emotion_embed_hard is not None else emotion_embed_soft
            output = output + results["emo_emb_1"].unsqueeze(1).expand(-1, max_src_len, -1)

        if hasattr(self, "emotion_classifier_1"):
            results["emo_cls_1_output"] = self.emotion_classifier_1(results["emo_emb_1"])

        if hasattr(self, "speaker_classifier_1"):
            results["spk_cls_1_output"] = self.speaker_classifier_1(results["spk_emb_1"])

        if hasattr(self, "revgrad_1"):
            results["emo_cls_1_rev_output"] = self.revgrad_1(results["spk_emb_1"])

        (
            output,
            results["p_preds"],
            results["e_preds"],
            results["d_preds"],
            results["d_rounded"],
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            inputs["pitches"],
            inputs["energies"],
            inputs["durations"],
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)

        if hasattr(self, "speaker_encoder_2"):
            results["spk_emb_2"] = self.speaker_encoder_2(inputs["speakers"], inputs["spker_embeds"])
            output = output + results["spk_emb_2"].unsqueeze(1).expand(-1, max_mel_len, -1)

        if hasattr(self, "emotion_encoder_2"):
            emotion_embed_hard, emotion_embed_soft, score_hard, score_soft = self.emotion_encoder_2(
                inputs["mels"], inputs["emotions"])
            # 尽量用hard模式，选择对应某个情感的token，做为输入，而不是用reference的语音得到的soft token
            results["emo_emb_2"] = emotion_embed_hard if emotion_embed_hard is not None else emotion_embed_soft
            output = output + results["emo_emb_2"].unsqueeze(1).expand(-1, max_mel_len, -1)

        if hasattr(self, "emotion_classifier_2"):
            results["emo_cls_2_output"] = self.emotion_classifier_2(results["emo_emb_2"])

        if hasattr(self, "speaker_classifier_2"):
            results["spk_cls_2_output"] = self.speaker_classifier_2(results["spk_emb_2"])

        if hasattr(self, "revgrad_2"):
            results["emo_cls_2_rev_output"] = self.revgrad_2(results["spk_emb_2"])

        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output

        results["output"] = output
        results["postnet_output"] = postnet_output
        results["mel_lens"], results["src_lens"] = mel_lens, src_lens
        results["mel_masks"], results["src_masks"] = mel_masks, src_masks

        return results


def test():
    import yaml

    # model_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LibriTTS_emovDB/model.yaml"
    # train_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LibriTTS_emovDB/train.yaml"
    # preprocess_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LibriTTS_emovDB/preprocess.yaml"

    model_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/EmovDB/model_test2.yaml"
    train_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/EmovDB/train.yaml"
    preprocess_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/EmovDB/preprocess.yaml"


    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastSpeech2(preprocess_config, model_config)
    model.to(device)
    model.train()
    # print(model)

    batch_size = 4
    max_src_len = 10
    max_mel_len = 40
    mel_dim = 80
    embed_dim = 256
    ids = torch.randint(0, 100, (batch_size,)).to(device)
    raw_texts = torch.randint(0, 100, (batch_size, max_src_len)).to(device)
    speakers = torch.randint(0, 4, (batch_size, )).to(device)
    emotions = torch.randint(0, 8, (batch_size, )).to(device)
    texts = torch.randint(0, 10, (batch_size, max_src_len)).to(device)
    src_lens = torch.randint(max_src_len, max_src_len+1, (batch_size, )).to(device)
    mels = torch.rand((batch_size, max_mel_len, mel_dim)).float().to(device)
    mel_lens = torch.randint(max_mel_len, max_mel_len+1, (batch_size, )).long().to(device)
    p_targets = torch.randint(0, 10, (batch_size, max_src_len)).float().to(device)
    e_targets = torch.randint(0, 10, (batch_size, max_src_len)).float().to(device)
    d_targets = torch.randint(0, 10, (batch_size, max_src_len)).float().to(device)
    spker_embeds = torch.rand((batch_size, 192)).float().to(device)

    x = defaultdict(
        lambda: None, {
            "ids": ids,
            "raw_texts": raw_texts,
            "speakers": speakers,
            "emotions": emotions,
            "texts": texts,
            "src_lens": src_lens,
            "max_src_len": max(src_lens),
            "mels": mels,
            "mel_lens": mel_lens,
            "max_mel_len": max(mel_lens),
            "pitches": p_targets,
            "energies": e_targets,
            "durations": d_targets,
            "spker_embeds": spker_embeds,
        })

    for k,v in model.named_parameters():
        print(k, v.shape)

    print()

    for k, v in x.items():
        print(k, v.shape)

    y = model(x)

    # print(emotion_classifier_1_output.shape)
    # print(emotion_classifier_1_output)
    # print(type(emotion_classifier_1_output))
    # print(emotions.shape)

    # for item in y:
    #     print(item.shape)

    loss = model.criterion(x, y)



    for k,v in y.items():
        print(k, v.shape)
    # print(y.keys())
    print()

    for k,v in loss.items():
        print(k,":", v)

    # (
    #     total_loss,
    #     mel_loss,
    #     postnet_mel_loss,
    #     pitch_loss,
    #     energy_loss,
    #     duration_loss,
    #     speaker_loss_1,
    #     speaker_loss_2,
    #     emotion_loss_1,
    #     emotion_loss_2,
    #     emotion_loss_1_revgrad,
    #     emotion_loss_2_revgrad,
    #     speaker_emotion_loss_1,
    #     speaker_emotion_loss_2,
    #     emotion_style_loss,
    #     loss_1,
    #     loss_2,
    #     all_loss,
    # ) = loss

    # print(f"total_loss: {total_loss}")
    # print(f"mel_loss: {mel_loss}")
    # print(f"postnet_mel_loss: {postnet_mel_loss}")
    # print(f"pitch_loss: {pitch_loss}")
    # print(f"energy_loss: {energy_loss}")
    # print(f"duration_loss: {duration_loss}")
    # print(f"speaker_loss_1: {speaker_loss_1}")
    # print(f"speaker_loss_2: {speaker_loss_2}")
    # print(f"emotion_loss_1: {emotion_loss_1}")
    # print(f"emotion_loss_2: {emotion_loss_2}")
    # print(f"emotion_loss_1_revgrad: {emotion_loss_1_revgrad}")
    # print(f"emotion_loss_2_revgrad: {emotion_loss_2_revgrad}")
    # print(f"speaker_emotion_loss_1: {speaker_emotion_loss_1}")
    # print(f"speaker_emotion_loss_2: {speaker_emotion_loss_2}")
    # print(f"emotion_style_loss: {emotion_style_loss}")
    # print(f"loss_1: {loss_1}")
    # print(f"loss_2: {loss_2}")
    # print(f"all_loss: {all_loss}")




if __name__ == "__main__":
    test()
