import torch
import config as cfg
from brain_Transformer import ActorNetwork
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Create a colormap from gray to red
colors = ["gray", "red"]  # Start with gray and transition to red
cmap_name = "gray_to_red"
n_bins = 100  # Increase this number for a smoother transition
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

model = ActorNetwork(cfg.input_size)

model_path = 'TrainingResult\\20240507-2Head\\param\\actor_torch_ppo.pt'
# param_path = os.path.join(cfg.experi_dir(), 'param')
# model_path = os.path.join(param_path, 'actor_torch_ppo.pt')
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

"""viz neuron weights"""
# for name, param in model.named_parameters():
#     if 'weight' in name:  # This checks for weights in linear layers
#         print(f"{name}: {param.size()}")
#         weights = param.detach().numpy()
#
#         # Assuming `weights` is a 2D numpy array from one of the layers
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(weights, cmap='viridis')
#         plt.title('Heatmap of Weights')
#         plt.xlabel('Features')
#         plt.ylabel('Neurons')
#         plt.show()
#
#         # For histograms
#         plt.hist(weights.flatten(), bins=50, alpha=0.7)
#         plt.title('Distribution of Weight Values')
#         plt.show()

"""viz attention score"""
matrix_file = os.path.join(cfg.experi_dir(), 'state_matrix.txt')
state = np.loadtxt(matrix_file)
state = np.array([state])
state_tensor = torch.tensor(state, dtype=torch.float)
dis = model(state_tensor)
print(model.attn_weights_1.shape)

l1_head_1 = model.attn_weights_1[:, 0, :, :].detach().cpu().numpy()
l1_head_2 = model.attn_weights_1[:, 1, :, :].detach().cpu().numpy()

l2_head_1 = model.attn_weights_2[:, 0, :, :].detach().cpu().numpy()
l2_head_2 = model.attn_weights_2[:, 1, :, :].detach().cpu().numpy()

plt.figure(figsize=(10, 8))
sns.heatmap(l1_head_1[0], annot=False, cmap='PiYG')
plt.title('Actor Attention Weights - First Head of First Layer')
plt.xlabel('Key/Sequence Position')
plt.ylabel('Query/Sequence Position')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(l1_head_2[0], annot=False, cmap='PiYG')
plt.title('Actor Attention Weights - Second Head of First Layer')
plt.xlabel('Key/Sequence Position')
plt.ylabel('Query/Sequence Position')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(l2_head_1[0], annot=False, cmap='PiYG')
plt.title('Actor Attention Weights - First Head of Second Layer')
plt.xlabel('Key/Sequence Position')
plt.ylabel('Query/Sequence Position')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(l2_head_2[0], annot=False, cmap='PiYG')
plt.title('Actor Attention Weights - Second Head of Second Layer')
plt.xlabel('Key/Sequence Position')
plt.ylabel('Query/Sequence Position')
plt.show()
