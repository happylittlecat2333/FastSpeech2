import os
import json
from typing import Text
from matplotlib.pyplot import text

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)

# from speaker_model import SpeakerEmbedding
# from model.speaker_embedding import SpeakerEmbedding
from model.modules import VarianceAdaptor, VAE, SpeakerEmbedding, SpeakerClassifier, EmotionClassifier
from model.modules import GradientReversal, PostNet, StyleEncoder, GlobalEmotionToken
from model.loss import FastSpeech2Loss
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """
    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        if model_config["block_type"] == "transformer":
            from model.transformers.transformer import TextEncoder, Decoder
        elif model_config["block_type"] == "lstransformer":
            from model.transformers.lstransformer import TextEncoder, Decoder
        elif model_config["block_type"] == "fastformer":
            from model.transformers.fastformer import TextEncoder, Decoder
        elif model_config["block_type"] == "conformer":
            from model.transformers.conformer import TextEncoder, Decoder
        elif model_config["block_type"] == "reformer":
            from model.transformers.reformer import TextEncoder, Decoder
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
                    if model_config["Encoder_config"]["use_revgrad"]:
                        self.revgrad_1 = nn.Sequential(
                            GradientReversal(),
                            EmotionClassifier(preprocess_config, model_config),
                        )

            if model_config["Decoder_config"]["use_speaker"]:
                self.speaker_encoder_2 = SpeakerEmbedding(preprocess_config, model_config)
                if model_config["Decoder_config"]["use_speaker_classifier"]:
                    self.speaker_classifier_2 = SpeakerClassifier(preprocess_config, model_config)
                    if model_config["Decoder_config"]["use_revgrad"]:
                        self.revgrad_2 = nn.Sequential(
                            GradientReversal(),
                            EmotionClassifier(preprocess_config, model_config),
                        )

        if model_config["multi_emotion"]:
            if model_config["Encoder_config"]["use_emotion"]:
                self.emotion_encoder_1 = GlobalEmotionToken(preprocess_config, model_config)
                if model_config["Encoder_config"]["use_emotion_classifier"]:
                    self.emotion_classifier_1 = EmotionClassifier(preprocess_config, model_config)     

            if model_config["Decoder_config"]["use_emotion"]:
                self.emotion_encoder_2 = GlobalEmotionToken(preprocess_config, model_config)
                if model_config["Decoder_config"]["use_emotion_classifier"]:
                    self.emotion_classifier_2 = EmotionClassifier(preprocess_config, model_config)

        self.criterion = FastSpeech2Loss(preprocess_config, model_config)

    def forward(
        self,
        speakers,
        emotions,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        spker_embeds=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None)

        output, src_word_emb = self.encoder(texts, src_masks)

        speaker_emb_1, speaker_emb_2 = None, None
        emotion_emb_1, emotion_emb_2 = None, None

        emotion_classifier_1_output = None
        emotion_classifier_2_output = None
        speaker_classifier_1_output = None
        speaker_classifier_2_output = None
        emotion_classifier_1_revgrad_output = None
        emotion_classifier_2_revgrad_output = None

        if hasattr(self, "speaker_encoder_1"):
            # speaker_emb_1 = self.speaker_encoder_1(mels, mel_lens)
            speaker_emb_1 = self.speaker_encoder_1(speakers, spker_embeds)
            output = output + speaker_emb_1.unsqueeze(1).expand(-1, max_src_len, -1)

        if hasattr(self, "emotion_encoder_1"):
            # emotion_emb_1 = self.emotion_encoder_1(mels, emotions)
            emotion_embed_hard, emotion_embed_soft, score_hard, score_soft = self.emotion_encoder_1(mels, emotions)
            # 尽量用hard模式，选择对应某个情感的token，做为输入，而不是用reference的语音得到的soft token
            if self.training:
                # print("HARD emotion token")
                output = output + emotion_embed_hard.expand(
                    -1, max_src_len, -1
                )
            elif emotion_embed_hard is not None:
                # print("HARD emotion token")
                output = output + emotion_embed_hard.expand(
                    -1, max_src_len, -1
                )
            else:
                # print("SOFT emotion token")
                output = output + emotion_embed_soft.expand(
                    -1, max_src_len, -1
                )

            # output = output + emotion_emb_1.unsqueeze(1).expand(-1, max_src_len, -1)

        if hasattr(self, "emotion_classifier_1"):
            emotion_classifier_1_output = self.emotion_classifier_1(emotion_emb_1)

        if hasattr(self, "speaker_classifier_1"):
            speaker_classifier_1_output = self.speaker_classifier_1(speaker_emb_1)

        if hasattr(self, "revgrad_1"):
            emotion_classifier_1_revgrad_output = self.revgrad_1(speaker_emb_1)


        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        if hasattr(self, "speaker_encoder_2"):
            # speaker_emb_2 = self.speaker_encoder_2(postnet_output, mel_lens)
            speaker_emb_2 = self.speaker_encoder_2(speakers, spker_embeds)

        if hasattr(self, "emotion_encoder_2"):
            emotion_emb_2 = self.emotion_encoder_2(postnet_output, emotions)

        if hasattr(self, "emotion_classifier_2"):
            emotion_classifier_2_output = self.emotion_classifier_2(emotion_emb_2)

        if hasattr(self, "speaker_classifier_2"):
            speaker_classifier_2_output = self.speaker_classifier_2(speaker_emb_2)

        if hasattr(self, "revgrad_2"):
            emotion_classifier_2_revgrad_output = self.revgrad_2(speaker_emb_2)

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            speaker_emb_1,
            speaker_emb_2,
            emotion_emb_1,
            emotion_emb_2,
            emotion_classifier_1_output,
            emotion_classifier_2_output,
            speaker_classifier_1_output,
            speaker_classifier_2_output,
            emotion_classifier_1_revgrad_output,
            emotion_classifier_2_revgrad_output
        )


def test():
    import yaml
    # model_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LibriTTS_emovDB/model.yaml"
    # train_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LibriTTS_emovDB/train.yaml"
    # preprocess_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LibriTTS_emovDB/preprocess.yaml"

    model_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LJSpeech/model.yaml"
    train_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LJSpeech/train.yaml"
    preprocess_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LJSpeech/preprocess.yaml"


    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastSpeech2(preprocess_config, model_config)
    model.to(device)
    model.train()
    print(model)

    batch_size = 16
    max_src_len = 100
    max_mel_len = 200
    mel_dim = 80
    ids = torch.randint(0, 100, (batch_size,)).to(device)
    raw_texts = torch.randint(0, 100, (batch_size, max_src_len)).to(device)
    speakers = torch.randint(0, 4, (batch_size, )).to(device)
    emotions = torch.randint(0, 8, (batch_size, )).to(device)
    texts = torch.randint(0, 10, (batch_size, max_src_len)).to(device)
    src_lens = torch.randint(1, max_src_len, (batch_size, )).to(device)
    mels = torch.rand((batch_size, max_mel_len, mel_dim)).float().to(device)
    mel_lens = torch.randint(1, max_mel_len, (batch_size, )).long().to(device)
    p_targets = torch.randint(0, 10, (batch_size, max_src_len)).float().to(device)
    e_targets = torch.randint(0, 10, (batch_size, max_src_len)).float().to(device)
    d_targets = torch.randint(0, 10, (batch_size, max_src_len)).float().to(device)

    x = (
        ids,
        raw_texts,
        speakers,
        emotions,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets,
        e_targets,
        d_targets,
    )

    y = model(*x[2:])

    (
        output,
        postnet_output,
        p_predictions,
        e_predictions,
        log_d_predictions,
        d_rounded,
        src_masks,
        mel_masks,
        src_lens,
        mel_lens,
        speaker_emb_1,
        speaker_emb_2,
        emotion_emb_1,
        emotion_emb_2,
        emotion_classifier_1_output,
        emotion_classifier_2_output,
        speaker_classifier_1_output,
        speaker_classifier_2_output,
        emotion_classifier_1_revgrad_output,
        emotion_classifier_2_revgrad_output
    ) = y

    # print(emotion_classifier_1_output.shape)
    # print(emotion_classifier_1_output)
    # print(type(emotion_classifier_1_output))
    # print(emotions.shape)

    # for item in y:
    #     print(item.shape)

    loss = model.criterion(x, y)

    (
        total_loss,
        mel_loss,
        postnet_mel_loss,
        pitch_loss,
        energy_loss,
        duration_loss,
        speaker_loss_1,
        speaker_loss_2,
        emotion_loss_1,
        emotion_loss_2,
        emotion_loss_1_revgrad,
        emotion_loss_2_revgrad,
        speaker_emotion_loss_1,
        speaker_emotion_loss_2,
        emotion_style_loss,
        loss_1,
        loss_2,
        all_loss,
    ) = loss

    print(f"total_loss: {total_loss}")
    print(f"mel_loss: {mel_loss}")
    print(f"postnet_mel_loss: {postnet_mel_loss}")
    print(f"pitch_loss: {pitch_loss}")
    print(f"energy_loss: {energy_loss}")
    print(f"duration_loss: {duration_loss}")
    print(f"speaker_loss_1: {speaker_loss_1}")
    print(f"speaker_loss_2: {speaker_loss_2}")
    print(f"emotion_loss_1: {emotion_loss_1}")
    print(f"emotion_loss_2: {emotion_loss_2}")
    print(f"emotion_loss_1_revgrad: {emotion_loss_1_revgrad}")
    print(f"emotion_loss_2_revgrad: {emotion_loss_2_revgrad}")
    print(f"speaker_emotion_loss_1: {speaker_emotion_loss_1}")
    print(f"speaker_emotion_loss_2: {speaker_emotion_loss_2}")
    print(f"emotion_style_loss: {emotion_style_loss}")
    print(f"loss_1: {loss_1}")
    print(f"loss_2: {loss_2}")
    print(f"all_loss: {all_loss}")




if __name__ == "__main__":
    test()