import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import model

from model.modules import StyleLoss, OrthogonalLoss




class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.style_loss = StyleLoss()
        self.ort_loss = OrthogonalLoss()


    def forward(self, inputs, predictions):
        (
            ids,
            raw_texts,
            speakers,
            emotions,
            texts,
            text_lens,
            max_text_len,
            mel_targets,
            mel_lens,
            max_mel_len,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
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
        ) = predictions

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)

        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        Loss = dict()

        speaker_loss_1, speaker_loss_2 = torch.tensor(0.), torch.tensor(0.)
        emotion_loss_1, emotion_loss_2 = torch.tensor(0.), torch.tensor(0.)
        emotion_loss_1_revgrad, emotion_loss_2_revgrad = torch.tensor(0.), torch.tensor(0.)
        speaker_emotion_loss_1, speaker_emotion_loss_2 = torch.tensor(0.), torch.tensor(0.)
        emotion_style_loss = torch.tensor(0.)

        # emotion_targets = F.one_hot(emotions, num_classes=self.emotion_num).float()
        # speaker_targets = F.one_hot(speakers, num_classes=self.speaker_num).float()
        emotion_targets = emotions
        speaker_targets = speakers
        # print(emotion_targets.shape)
        # print(emotion_classifier_1_output.shape)

        if self.model_config["Encoder_config"]["use_speaker"]:
            Loss["speaker_loss_1"] = self.ce_loss(speaker_classifier_1_output, speaker_targets)
        if self.model_config["Decoder_config"]["use_speaker"]:
            Loss["speaker_loss_2"] = self.ce_loss(speaker_classifier_2_output, speaker_targets)

        if self.model_config["Encoder_config"]["use_emotion"]:
            Loss["emotion_loss_1"] = self.ce_loss(emotion_classifier_1_output, emotion_targets)
        if self.model_config["Decoder_config"]["use_emotion"]:
            Loss["emotion_loss_2"] = self.ce_loss(emotion_classifier_2_output, emotion_targets)

        # print(emotion_loss_1.shape)
        # print(emotion_loss_1)

        if self.model_config["Encoder_config"]["use_revgrad"]:
            Loss["emotion_loss_1_revgrad"] = self.ce_loss(emotion_classifier_1_revgrad_output, emotion_targets)
        if self.model_config["Decoder_config"]["use_revgrad"]:
            Loss["emotion_loss_2_revgrad"] = self.ce_loss(emotion_classifier_2_revgrad_output, emotion_targets)

        if self.model_config["Loss_config"]["use_orthogonal_loss"]:
            Loss["speaker_emotion_loss_1"] = self.ort_loss(speaker_emb_1, emotion_emb_1)
            Loss["speaker_emotion_loss_2"] = self.ort_loss(speaker_emb_2, emotion_emb_2)

        if self.model_config["Loss_config"]["use_style_loss"]:
            Loss["emotion_style_loss"] = self.style_loss(emotion_emb_1, emotion_emb_2)

        Loss["loss_1"] = (
            speaker_loss_1 + emotion_loss_1 + speaker_emotion_loss_1 + emotion_loss_1_revgrad
        )
        Loss["loss_2"] = (
            speaker_loss_2 + emotion_loss_2 + speaker_emotion_loss_2 + emotion_loss_2_revgrad
        )

        Loss["mel_loss"] = self.mae_loss(mel_predictions, mel_targets)
        Loss["postnet_mel_loss"] = self.mae_loss(postnet_mel_predictions, mel_targets)

        Loss["pitch_loss"] = self.mse_loss(pitch_predictions, pitch_targets)
        Loss["energy_loss"] = self.mse_loss(energy_predictions, energy_targets)
        Loss["duration_loss"] = self.mse_loss(log_duration_predictions, log_duration_targets)

        Loss["all_loss"] = Loss["mel_loss"] + Loss["postnet_mel_loss"] + Loss["pitch_loss"] + Loss["energy_loss"]

        Loss["total_loss"] = Loss["all_loss"] + Loss["loss_1"] + Loss["loss_2"] + Loss["emotion_style_loss"]
                

        return Loss

        # return (
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
        # )
