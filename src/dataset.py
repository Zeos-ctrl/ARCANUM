import numpy as np
import torch
from torch.utils.data import Dataset
class GWFlatDataset(Dataset):
    """
    Flattened dataset that yields one time-sample at a time:
    For idx in [0 .. num_samples * N_common - 1]:
      i = idx // N_common   (which waveform)
      j = idx % N_common    (which time index)

    Input: (t_norm[j], theta_norm_all[i, :]) → shape (6,)
    Targets: (amp_norm[i, j], dphi[i, j])     → shape (2,)
    """

    def __init__(self, waveform_chunks, theta_norm_all, time_norm, N_common):
        """
        Args:
          waveform_chunks:   list of length num_samples, each a dict with keys
                             "start_idx", "amp_chunk", "dphi_chunk"
          theta_norm_all:    numpy array of shape (num_samples, 5)
          time_norm:         numpy array of length N_common (float32)
          N_common:          int, number of time-samples per waveform
        """
        self.waveform_chunks = waveform_chunks
        self.theta_norm_all  = theta_norm_all  # now (N,5)
        self.time_norm       = time_norm
        self.N_common        = N_common
        self.num_samples     = theta_norm_all.shape[0]
        self.length          = self.num_samples * self.N_common

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # figure out which waveform i and time-index j
        i = idx // self.N_common
        j = idx % self.N_common

        # build input vector of shape (6,)
        #   [ t_norm[j], M, eta, chi_eff, chi_p, ecc ]
        t_j     = self.time_norm[j]               # scalar float32
        theta_i = self.theta_norm_all[i, :]       # (5,)
        x       = np.empty(6, dtype=np.float32)
        x[0]    = t_j
        x[1:]   = theta_i
        x_tensor = torch.from_numpy(x).to(self.DEVICE)

        # now fetch amplitude and phase-rate
        chunk = self.waveform_chunks[i]
        start_idx  = chunk["start_idx"]
        amp_chunk  = chunk["amp_chunk"]
        dphi_chunk = chunk["dphi_chunk"]
        chunk_len  = amp_chunk.shape[0]

        # if j is inside the stored chunk, grab that sample, else zeros
        if start_idx <= j < start_idx + chunk_len:
            local = j - start_idx
            amp_val, dphi_val = amp_chunk[local], dphi_chunk[local]
        else:
            amp_val, dphi_val = 0.0, 0.0

        y = np.array([amp_val, dphi_val], dtype=np.float32)
        y_tensor = torch.from_numpy(y).to(self.DEVICE)

        return x_tensor, y_tensor

