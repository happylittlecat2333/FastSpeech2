# %%
import yaml
from yaml import loader

path = "/home/xjl/Audio/Library/Models/MyFastSpeech2/speaker_model/pretrained_models/spkrec-ecapa-voxceleb/hyperparams.yaml"

config = yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


# %%
import numpy as np

p = "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/LJSpeech_v2/mel/LJSpeech-neutral-mel-LJSpeech_neutral_LJ001-0001.npy"

x = np.load(p)

x.shape
# %%

p = "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/EmovDB_v2/duration/bea-amused-duration-bea_amused_amused_1-15_0001.npy"
np.load(p).shape
# %%

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x = np.random.rand(100, 10, 80)

for i in x:
    # i = i.reshape(-1, 1)
    scaler.partial_fit(i)

# %%
p = "/home/xjl/Audio/Library/Models/MyFastSpeech2/dump/EmovDB/stats/stats.npy"
np.load(p).shape

# %%
t = x[0]
print(t.shape)
tt = (t-scaler.mean_) / scaler.scale_
print(tt.shape)
tt
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x = np.random.rand(100, 10, 1)

for i in x:
    # i = i.reshape(-1, 1)
    scaler.partial_fit(i)
# %%

t = x[0]
print(t.shape)
tt = (t - scaler.mean_) / scaler.scale_
print(tt.shape)
tt
# %%
import numpy as np

p = [
    "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/EmovDB_v3/energy_frame/sam-amused-energy-sam_amused_amused_1-28_0001.npy",
    "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/EmovDB_v3/mel/bea-amused-mel-bea_amused_amused_1-15_0001.npy",
    "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/EmovDB_v3/pitch_frame/sam-amused-pitch-sam_amused_amused_1-28_0001.npy",
]
p2 = [
    "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/EmovDB_v2/energy_frame/sam-amused-energy-sam_amused_amused_1-28_0001.npy",
    "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/EmovDB_v2/mel/bea-amused-mel-bea_amused_amused_1-15_0001.npy",
    "/home/xjl/Audio/Library/Models/MyFastSpeech2/preprocessed_data/EmovDB_v2/pitch_frame/sam-amused-pitch-sam_amused_amused_1-28_0001.npy",
]

for i in range(len(p)):
    x = np.load(p[i])
    x2 = np.load(p2[i])
    print(x.shape, x2.shape)

# a,b = np.load(p[i]), np.load(p2[i])

# %%
i = 1
a, b = np.load(p[i]), np.load(p2[i])
# %%
