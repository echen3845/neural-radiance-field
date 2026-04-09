import os
import torch

from dataset import BlenderNeRFDataset
from encoding import PositionalEncoding
from model import NeRF
from train_scene import render_orbit_frames, save_gif  # or move these helpers elsewhere

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset metadata for image size / focal length
ds = BlenderNeRFDataset("data/lego", split="train")
sample = ds[0]
H, W, focal = sample["H"], sample["W"], sample["focal"]

# Render settings must match training reasonably closely
near = 2.0
far = 6.0
n_samples = 64

# IMPORTANT: match checkpoint architecture
pos_encoder = PositionalEncoding(input_dims=3, num_freqs=10).to(device)
dir_encoder = PositionalEncoding(input_dims=3, num_freqs=4).to(device)

model = NeRF(
    pos_in_dims=pos_encoder.out_dim,
    dir_in_dims=dir_encoder.out_dim
).to(device)

checkpoint = torch.load(
    "checkpoints/nerf_lego_step_Coarse + Fine 100k 64 128 2048 no scheduler.pt",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from step:", checkpoint["step"])

os.makedirs("outputs", exist_ok=True)

orbit_frames = render_orbit_frames(
    model=model,
    pos_encoder=pos_encoder,
    dir_encoder=dir_encoder,
    H=H,
    W=W,
    focal=focal,
    near=near,
    far=far,
    n_samples=n_samples,
    device=device,
    n_frames=30,
    radius=4.0,
    phi_deg=-30.0,
)

save_gif(orbit_frames, "outputs/lego.gif", fps=20)