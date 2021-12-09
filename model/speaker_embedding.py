# %%

import torch
import torch.nn as nn
import yaml

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
# from deepspeaker import embedding



class PreDefinedEmbedder(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(PreDefinedEmbedder, self).__init__()
        self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]

        self.mel_dim = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.win_length = preprocess_config["preprocessing"]["stft"]["win_length"]

        self.pretrained_path = model_config["speaker_embedding"]["pretrained_model"][self.embedder_type]["pretrained_path"]
        self.config_path = model_config["speaker_embedding"]["pretrained_model"][self.embedder_type]["config_path"]

        self.embedder = self._get_speaker_embedder()


    def _get_speaker_embedder(self):

        embedder = None

        if self.embedder_type == "ECAPA-TDNN":
            embedding_model = yaml.load(open(self.config_path, 'r'), Loader=yaml.FullLoader)["embedding_model"]
            embedder = self._get_ecapa_tdnn(self.pretrained_path, embedding_model)

        # if self.embedder_type == "DeepSpeaker":
        #     embedder = embedding.build_model(
        #         "./deepspeaker/pretrained_models/ResCNN_triplet_training_checkpoint_265.h5"
        #     )

        else:
            raise NotImplementedError

        return embedder

    def forward(self, x):
        if self.embedder_type == "ECAPA-TDNN":
            x = torch.squeeze(self.embedder(x.unsqueeze_(0)))

        # if self.embedder_type == "DeepSpeaker":
        #     spker_embed = embedding.predict_embedding(self.embedder, x, self.sampling_rate, self.win_length,
        #                                               self.embedder_cuda)

        return x


    def _get_ecapa_tdnn(self, pretrained_path, embedding_model):
        ecapa_tdnn = ECAPA_TDNN(**embedding_model).eval()
        ecapa_tdnn.load_state_dict(torch.load(pretrained_path))
        for p in ecapa_tdnn.parameters():
            p.requires_grad = False
        return ecapa_tdnn


if __name__ == "__main__":

    import yaml
    model_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LJSpeech/model.yaml"
    train_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LJSpeech/train.yaml"
    preprocess_config_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/LJSpeech/preprocess.yaml"


    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)

    # ecapa_path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/config/EmovDB/model.yaml"
    # config = yaml.load(open(ecapa_path, "r"), Loader=yaml.FullLoader)
    # print(config)
    model = PreDefinedEmbedder(preprocess_config, model_config)
    input_feats = torch.rand([120, 80])
    print(input_feats.shape)
    output = model(input_feats)
    print(output.shape)
    # for p in model.parameters():
    #     if p.requires_grad:
    #         print(p.shape)

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
