import torch
import torch.nn as nn

from .soft_dtw_cuda_org import SoftDTW

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.start = train_config["loss"]["kl_start"]
        self.end = train_config["loss"]["kl_end"]
        self.upper = train_config["loss"]["kl_upper"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.sdtw_loss = SoftDTW(
            use_cuda=True,
            gamma=train_config["loss"]["gamma"],
            # warp=train_config["loss"]["warp"],
        )
        # self.L = model_config["decoder"]["decoder_layer"]

    def kl_anneal(self, step):
        if step < self.start:
            return .0
        elif step >= self.end:
            return self.upper
        else:
            return self.upper*((step - self.start) / (self.end - self.start))

    def forward(self, inputs, predictions, step):
        (
            src_lens_targets,
            _,
            mel_targets,
            mel_lens_targets,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[4:]
        (
            mel_iters,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            mus,
            log_vars,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        src_lens_targets.requires_grad = False
        mel_lens_targets.requires_grad = False
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

        # mel_predictions = mel_iters[-1].masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # Iterative Loss Using Soft-DTW
        mel_iter_loss = 0
        for mel_iter in mel_iters:
            mel_iter_loss += self.mae_loss(mel_iter.masked_select(mel_masks.unsqueeze(-1)), mel_targets)
        mel_loss = mel_iter_loss / len(mel_iters)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        # postnet_mel_loss = torch.tensor([0.], device=mel_targets.device)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # KL Divergence Loss
        beta = torch.tensor(self.kl_anneal(step))
        kl_loss = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + beta * kl_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            kl_loss,
            beta,
        )


# import torch
# import torch.nn as nn

# from .soft_dtw_cuda_org import SoftDTW

# class FastSpeech2Loss(nn.Module):
#     """ FastSpeech2 Loss """

#     def __init__(self, preprocess_config, model_config, train_config):
#         super(FastSpeech2Loss, self).__init__()
#         self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
#             "feature"
#         ]
#         self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
#             "feature"
#         ]
#         self.start = train_config["loss"]["kl_start"]
#         self.end = train_config["loss"]["kl_end"]
#         self.upper = train_config["loss"]["kl_upper"]
#         self.mse_loss = nn.MSELoss()
#         self.mae_loss = nn.L1Loss()
#         self.sdtw_loss = SoftDTW(
#             use_cuda=True,
#             gamma=train_config["loss"]["gamma"],
#             # warp=train_config["loss"]["warp"],
#         )
#         # self.L = model_config["decoder"]["decoder_layer"]

#     def kl_anneal(self, step):
#         if step < self.start:
#             return .0
#         elif step >= self.end:
#             return self.upper
#         else:
#             return self.upper*((step - self.start) / (self.end - self.start))

#     def forward(self, inputs, predictions, step):
#         (
#             src_lens_targets,
#             _,
#             mel_targets,
#             mel_lens_targets,
#             _,
#             pitch_targets,
#             energy_targets,
#             duration_targets,
#         ) = inputs[4:]
#         (
#             mel_iters,
#             _,
#             pitch_predictions,
#             energy_predictions,
#             log_duration_predictions,
#             _,
#             src_masks,
#             mel_masks,
#             _,
#             _,
#             mus,
#             log_vars,
#             _,
#             _,
#         ) = predictions
#         src_masks = ~src_masks
#         mel_masks = ~mel_masks
#         log_duration_targets = torch.log(duration_targets.float() + 1)
#         mel_targets = mel_targets[:, : mel_masks.shape[1], :]
#         mel_masks = mel_masks[:, :mel_masks.shape[1]]

#         src_lens_targets.requires_grad = False
#         mel_lens_targets.requires_grad = False
#         log_duration_targets.requires_grad = False
#         pitch_targets.requires_grad = False
#         energy_targets.requires_grad = False
#         mel_targets.requires_grad = False

#         if self.pitch_feature_level == "phoneme_level":
#             pitch_predictions = pitch_predictions.masked_select(src_masks)
#             pitch_targets = pitch_targets.masked_select(src_masks)
#         elif self.pitch_feature_level == "frame_level":
#             pitch_predictions = pitch_predictions.masked_select(mel_masks)
#             pitch_targets = pitch_targets.masked_select(mel_masks)

#         if self.energy_feature_level == "phoneme_level":
#             energy_predictions = energy_predictions.masked_select(src_masks)
#             energy_targets = energy_targets.masked_select(src_masks)
#         if self.energy_feature_level == "frame_level":
#             energy_predictions = energy_predictions.masked_select(mel_masks)
#             energy_targets = energy_targets.masked_select(mel_masks)

#         log_duration_predictions = log_duration_predictions.masked_select(src_masks)
#         log_duration_targets = log_duration_targets.masked_select(src_masks)

#         mel_predictions = mel_iters[-1].masked_select(mel_masks.unsqueeze(-1))
#         # postnet_mel_predictions = postnet_mel_predictions.masked_select(
#         #     mel_masks.unsqueeze(-1)
#         # )
#         mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

#         # Iterative Loss Using Soft-DTW
#         mel_iter_loss = 0
#         # mel_iter_loss = torch.zeros_like(mel_lens_targets, dtype=mel_targets.dtype)
#         for mel_iter in mel_iters:
#             mel_iter_loss += self.mae_loss(mel_iter.masked_select(mel_masks.unsqueeze(-1)), mel_targets)
#             # mel_iter_loss += self.sdtw_loss(mel_iter, mel_targets)
#         mel_loss = mel_iter_loss / len(mel_iters)
#         # mel_loss = (mel_iter_loss / (len(mel_iters) * mel_lens_targets)).mean()
#         # print(mel_iter_loss.shape, (len(mel_iters) * mel_lens_targets).shape, (mel_iter_loss / (len(mel_iters) * mel_lens_targets)).shape)
#         # exit(0)
#         # postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
#         postnet_mel_loss = torch.tensor([0.], device=mel_targets.device)

#         pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
#         energy_loss = self.mse_loss(energy_predictions, energy_targets)
#         duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

#         # KL Divergence Loss
#         beta = torch.tensor(self.kl_anneal(step))
#         kl_loss = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())

#         total_loss = (
#             mel_loss + duration_loss + pitch_loss + energy_loss + beta * kl_loss
#         )

#         return (
#             total_loss,
#             mel_loss,
#             postnet_mel_loss,
#             pitch_loss,
#             energy_loss,
#             duration_loss,
#             kl_loss,
#             beta,
#         )
