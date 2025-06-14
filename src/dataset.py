import numpy as np
import torch
from torch.utils.data import Dataset

class GWFlatDataset(Dataset):
    """
    Flattened dataset that yields one time‐sample at a time:
    For idx in [0 .. num_samples * N_common - 1]:
      i = idx // N_common   (which waveform)
      j = idx % N_common    (which time index)
    Input: (t_norm[j], theta_norm_all[i, :]) → shape (16,)
    Targets: (amp_norm[i, j], dphi[i, j]) → shape (2,)
    """

    def __init__(self, waveform_chunks, theta_norm_all, time_norm, N_common):
        """
        Args:
          waveform_chunks: list of length num_samples, each a dict with keys
                           "start_idx", "amp_chunk", "dphi_chunk"
          theta_norm_all:  numpy array of shape (num_samples, 15)
          time_norm:       numpy array of length N_common (float32)
          N_common:        int, number of time‐samples per waveform
        """
        self.waveform_chunks = waveform_chunks
        self.theta_norm_all  = theta_norm_all
        self.time_norm       = time_norm
        self.N_common        = N_common
        self.num_samples     = theta_norm_all.shape[0]
        self.length          = self.num_samples * self.N_common

        # Determine device at runtime
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Determine which waveform and which time‐index
        i = idx // self.N_common
        j = idx % self.N_common

        # Input tensor: [ t_norm[j], theta_norm_all[i, : ] ] → shape (16,)
        t_j = self.time_norm[j]                             # scalar float32
        theta_i = self.theta_norm_all[i, :]                  # (15,)
        x = np.empty(16, dtype=np.float32)
        x[0] = t_j
        x[1:] = theta_i
        x_tensor = torch.from_numpy(x).to(self.DEVICE)       # (16,)

        # Get chunk info for waveform i
        chunk_dict = self.waveform_chunks[i]
        start_idx  = chunk_dict["start_idx"]
        amp_chunk  = chunk_dict["amp_chunk"]   # 1D np.float32
        dphi_chunk = chunk_dict["dphi_chunk"]  # 1D np.float32
        chunk_len  = amp_chunk.shape[0]

        # Determine if j falls within [start_idx, start_idx + chunk_len)
        if start_idx <= j < start_idx + chunk_len:
            local_idx = j - start_idx
            amp_val   = amp_chunk[local_idx]
            dphi_val  = dphi_chunk[local_idx]
        else:
            amp_val  = 0.0
            dphi_val = 0.0

        y = np.array([amp_val, dphi_val], dtype=np.float32)  # shape (2,)
        y_tensor = torch.from_numpy(y).to(self.DEVICE)       # (2,)

        return x_tensor, y_tensor

