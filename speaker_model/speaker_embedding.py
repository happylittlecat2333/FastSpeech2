# %%

import torch
import torch.nn as nn

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

# %%


class SpeakerEmbedding(nn.Module):
    def __init__(self, config):
        super(SpeakerEmbedding, self).__init__()
        self.mel_dim = config["speaker_embedding"]["embedding_model"]["input_size"]
        self.hidden_size = config["speaker_embedding"]["embedding_model"]["lin_neurons"]
        self.embedding_size = config["speaker_embedding"]["embedding_size"]
        self.pretrained_path = config["speaker_embedding"]["pretrained_path"]
        self.embedding_model = config["speaker_embedding"]["embedding_model"]

        self.ecapa_tdnn = self._get_ecapa_tdnn(self.pretrained_path, self.embedding_model)
        self.fc = nn.Linear(self.hidden_size, self.embedding_size)

    def forward(self, x, lengths=None):
        x = torch.squeeze(self.ecapa_tdnn(x, lengths))
        x = self.fc(x)
        return x

    def _get_ecapa_tdnn(self, pretrained_path, embedding_model):
        ecapa_tdnn = ECAPA_TDNN(**embedding_model)
        ecapa_tdnn.load_state_dict(torch.load(pretrained_path))
        for p in ecapa_tdnn.parameters():
            p.requires_grad = False
        return ecapa_tdnn



if __name__ == "__main__":

    import yaml

    ecapa_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/EmovDB/model.yaml"
    config = yaml.load(open(ecapa_path, "r"), Loader=yaml.FullLoader)
    print(config)
    model = SpeakerEmbedding(config)
    input_feats = torch.rand([5, 120, 80])
    output = model(input_feats)
    print(output.shape)
    for p in model.parameters():
        if p.requires_grad:
            print(p.shape)






# # %%

# model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
#                                                    savedir="pretrained_models/spkrec-ecapa-voxceleb")


# pretrained_dict = model.modules.state_dict()

# ecapa_tdnn = ECAPA_TDNN(input_size=80)

# model_dict = ecapa_tdnn.state_dict()

# # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# # model_dict.update(pretrained_dict)

# # ecapa_tdnn.load_state_dict(model_dict)

# # %%
# import torch

# input_feats = torch.rand([5, 120, 80])


# # %%
# import torch
# import numpy as np

# mel_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/EmovDB/mel/bea-amused-mel-bea_amused_amused_1-15_0001.npy"
# mel_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/EmovDB/mel/bea-amused-mel-bea_amused_amused_1-15_0002.npy"
# mel = np.load(mel_path)
# mel = torch.from_numpy(mel)

# print(mel.shape)

# mel = mel.unsqueeze(0)
# print(mel.shape)

# ecapa_tdnn.eval()
# output = ecapa_tdnn(mel)
# print(output.shape)

# # %%

# input_feats = torch.rand([5, 120, 80])
# ecapa_tdnn.train()
# output = ecapa_tdnn(input_feats)
# print(output.shape)

# # %%
# import torch
# m_dict = torch.load(
#     "/home/xjl/Audio/Library/Models/MyFastSpeech2/ECAPA_TDNN/pretrained_models/spkrec-ecapa-voxceleb/embedding_model.ckpt"
# )

# ecapa_config = config["speaker_embedding"]["embedding_model"]
# print(ecapa_config)
# m = ECAPA_TDNN(**ecapa_config)
# m.load_state_dict(m_dict)
# # new_dict = m.state_dict()
# # m_dict = {k: v for k, v in m_dict.items() if k in new_dict}
# # m.load_state_dict(new_dict.update(m_dict))


# # %%

# # %%
