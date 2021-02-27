import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np


class Classification(Dataset):
    def __init__(self, player_info=None, transform=None, predict_flag=None, predict_image=None):
        """
        Args:
            df_player_info (pandas): Dictionary with information about
            each player team and number.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df_player_info = player_info
        self.transform = transform
        self.predict_flag = predict_flag
        self.predict_image = predict_image

    def __len__(self):
        if self.predict_flag:
            return 1
        else:
            return len(self.df_player_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.predict_flag:
            image = self.predict_image
            label1 = 0
            label2 = 0
        else:
            image_path = self.df_player_info.iloc[idx, 0]
            image = io.imread(image_path)
            label1 = self.df_player_info.iloc[idx, 1]
            label2 = self.df_player_info.iloc[idx, 2]

        label1 = np.array([label1])
        label1 = label1.astype("float").reshape(-1, 1)

        label2 = np.array([label2])
        label2 = label2.astype("float").reshape(-1, 1)

        sample = {"images": image, "teams": label1, "players": label2}


        # Apply transformation
        if self.transform:
            sample = self.transform(sample)

        return sample



