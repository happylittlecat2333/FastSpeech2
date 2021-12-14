import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import model
from collections import defaultdict

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
        self.Encoder_config = self.model_config["Encoder_config"]
        self.Decoder_config = self.model_config["Decoder_config"]
        self.Loss_config = self.model_config["Loss_config"]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.style_loss = StyleLoss()
        self.ort_loss = OrthogonalLoss()


    def forward(self, inputs, predictions):
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
            
            predictions: defaultdict(lamda: None)
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

        Returns:
            Loss: defaultdict(lamda: torch.tensor(0.))
                - mel_loss, postnet_mel_loss
                - pitch_loss, energy_loss, duration_loss
                - speaker_loss_1, speaker_loss_2
                - emotion_loss_1, emotion_loss_2
                - speaker_emotion_loss_1, speaker_emotion_loss_2
                - emotion_style_loss
                - loss_1 = *_loss_1
                - loss_2 = * loss_1
                - all_loss = mel_loss + postnet_mel_loss + pitch_loss + energy_loss + duration_loss
                - total_loss = all_loss + loss_1 + loss_2 + emotion_style_loss
        """

        mel_masks, src_masks = predictions["mel_masks"], predictions["src_masks"]
        pitch_targets, pitch_predictions = inputs["pitches"], predictions["p_preds"]
        energy_targets, energy_predictions = inputs["energies"], predictions["e_preds"]
        duration_targets, log_duration_predictions = inputs["durations"], predictions["d_preds"]
        mel_targets, mel_predictions = inputs["mels"], predictions["output"]
        postnet_mel_predictions = predictions["postnet_output"]

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
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        Loss = defaultdict(lambda: torch.tensor(0.))

        Loss["mel_loss"] = self.mae_loss(mel_predictions, mel_targets)
        Loss["postnet_mel_loss"] = self.mae_loss(postnet_mel_predictions, mel_targets)

        Loss["pitch_loss"] = self.mse_loss(pitch_predictions, pitch_targets)
        Loss["energy_loss"] = self.mse_loss(energy_predictions, energy_targets)
        Loss["duration_loss"] = self.mse_loss(log_duration_predictions, log_duration_targets)

        if self.Encoder_config["use_speaker"]:
            if self.Encoder_config["use_speaker_classifier"]:
                spk_cls_1_output, speakers = predictions["spk_cls_1_output"], inputs["speakers"]
                assert spk_cls_1_output is not None and speakers is not None
                Loss["speaker_loss_1"] = self.ce_loss(spk_cls_1_output, speakers)

        if self.Decoder_config["use_speaker"]:
            if self.Decoder_config["use_speaker_classifier"]:
                spk_cls_2_output, speakers = predictions["spk_cls_2_output"], inputs["speakers"]
                assert spk_cls_2_output is not None and speakers is not None
                Loss["speaker_loss_2"] = self.ce_loss(spk_cls_2_output, speakers)

        if self.Encoder_config["use_emotion"]:
            if self.Encoder_config["use_emotion_classifier"]:
                emo_cls_1_output, emotions = predictions["emo_cls_1_output"], inputs["emotions"]
                assert emo_cls_1_output is not None and emotions is not None
                Loss["emotion_loss_1"] = self.ce_loss(emo_cls_1_output, emotions)

        if self.Decoder_config["use_emotion"]:
            if self.Decoder_config["use_emotion_classifier"]:
                emo_cls_2_output, emotions = predictions["emo_cls_2_output"], inputs["emotions"]
                assert emo_cls_2_output is not None and emotions is not None
                Loss["emotion_loss_2"] = self.ce_loss(emo_cls_2_output, emotions)

        if self.Encoder_config["use_speaker"] and self.Encoder_config["use_emotion"]:
            if self.Encoder_config["use_revgrad"]:
                emo_cls_1_rev_output, speakers = predictions["emo_cls_1_rev_output"], inputs["speakers"]
                assert emo_cls_1_rev_output is not None and speakers is not None
                Loss["emotion_loss_1_revgrad"] = self.ce_loss(emo_cls_1_rev_output, speakers)

            if self.Encoder_config["use_orthogonal_loss"]:
                spk_emb_1, emo_emb_1 = predictions["spk_emb_1"], predictions["emo_emb_1"]
                assert spk_emb_1 is not None and emo_emb_1 is not None
                Loss["speaker_emotion_loss_1"] = self.ort_loss(spk_emb_1, emo_emb_1)

        if self.Decoder_config["use_speaker"] and self.Decoder_config["use_emotion"]:
            if self.Decoder_config["use_revgrad"]:
                emo_cls_2_rev_output, speakers = predictions["emo_cls_2_rev_output"], inputs["speakers"]
                assert emo_cls_2_rev_output is not None and speakers is not None
                Loss["emotion_loss_2_revgrad"] = self.ce_loss(emo_cls_2_rev_output, speakers)

            if self.Decoder_config["use_orthogonal_loss"]:
                spk_emb_2, emo_emb_2 = predictions["spk_emb_2"], predictions["emo_emb_2"]
                assert spk_emb_2 is not None and emo_emb_2 is not None
                Loss["speaker_emotion_loss_2"] = self.ort_loss(spk_emb_2, emo_emb_2)

        if self.Loss_config["use_style_loss"]:
            if self.Encoder_config["use_emotion"] and self.Decoder_config["use_emotion"]:                
                emo_emb_1, emo_emb_2 = predictions["emo_emb_1"], predictions["emo_emb_2"]
                assert emo_emb_1 is not None and emo_emb_2 is not None
                Loss["emotion_style_loss"] = self.style_loss(emo_emb_1, emo_emb_2)


        Loss["loss_1"] = (
            Loss["speaker_loss_1"] +
            Loss["emotion_loss_1"] +
            Loss["speaker_emotion_loss_1"] +
            Loss["emotion_loss_1_revgrad"]
        )

        Loss["loss_2"] = (
            Loss["speaker_loss_2"] +
            Loss["emotion_loss_2"] +
            Loss["speaker_emotion_loss_2"] +
            Loss["emotion_loss_2_revgrad"]
        )

        Loss["all_loss"] = (
            Loss["mel_loss"] +
            Loss["postnet_mel_loss"] +
            Loss["pitch_loss"] +
            Loss["energy_loss"] +
            Loss["duration_loss"]
        )

        Loss["total_loss"] = (
            Loss["all_loss"] +
            Loss["loss_1"] +
            Loss["loss_2"] +
            Loss["emotion_style_loss"]
        )

        return Loss
