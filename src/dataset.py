import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


class BlenderNeRFDataset:
    def __init__(self, root_dir: str, split: str = "train", white_background: bool = True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.white_background = white_background

        meta_path = self.root_dir / f"transforms_{split}.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.camera_angle_x = float(meta["camera_angle_x"])
        self.frames = meta["frames"]

        self.images = []
        self.poses = []

        for frame in self.frames:
            file_path = self.root_dir / f"{frame['file_path']}.png"
            image = imageio.imread(file_path).astype(np.float32) / 255.0  # H, W, 4 or H, W, 3

            if image.shape[-1] == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                if self.white_background:
                    image = rgb * alpha + (1.0 - alpha)
                else:
                    image = rgb
            else:
                image = image[..., :3]

            pose = np.array(frame["transform_matrix"], dtype=np.float32)

            self.images.append(image)
            self.poses.append(pose)

        self.images = np.stack(self.images, axis=0)   # [N, H, W, 3]
        self.poses = np.stack(self.poses, axis=0)     # [N, 4, 4]

        self.images = torch.from_numpy(self.images)
        self.poses = torch.from_numpy(self.poses)

        self.num_images, self.H, self.W, _ = self.images.shape
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: int):
        return {
            "image": self.images[idx],         # [H, W, 3]
            "pose": self.poses[idx],           # [4, 4]
            "H": self.H,
            "W": self.W,
            "focal": self.focal,
        }