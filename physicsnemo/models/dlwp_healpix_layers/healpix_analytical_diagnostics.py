import numpy as np
import torch
import xarray as xr

class DeriveTCWV(torch.nn.Module):
    def __init__(
        self,
        hPa_levels,
        output_channels,
        scaling,
        g0: float = 9.81,
    ):
        super().__init__()

        self.hPa_levels = hPa_levels
        self.output_channels = output_channels
        self.scaling = scaling
        self.g0 = g0

        if 'tcwv' in output_channels:
            raise ValueError('tcwv already in output channels list.')

        for hPa_level in self.hPa_levels:
            if f'q{hPa_level}' not in output_channels:
                raise ValueError(f'q{hPa_level} not found in output channels list, cannot derive tcwv.')

        if 'sp' not in output_channels:
            raise ValueError('sp not found in output channels list, cannot derive tcwv.')

        if not all(self.hPa_levels[i] < self.hPa_levels[i+1] for i in range(len(self.hPa_levels)-1)):
            raise ValueError('hPa_levels must be in increasing order (lowest to highest).')

        q_index = torch.tensor(
            [channels.index(f'q{hPa_level}') for hPa_level in self.hPa_levels],
            dtype=torch.int
        )
        sp_index = torch.tensor(output_channels.index('sp'), dtype=torch.int)
        tcwv_index = torch.tensor(output_channels.index('tcwv'), dtype=torch.int)

        q_mean = torch.tensor([scaling[f'q{hPa_level}']['mean'] for hPa_level in self.hPa_levels])
        q_std = torch.tensor([scaling[f'q{hPa_level}']['std'] for hPa_level in self.hPa_levels])
        sp_mean = torch.tensor(scaling['sp']['mean'])
        sp_std = torch.tensor(scaling['sp']['std'])
        tcwv_mean = torch.tensor(scaling['tcwv']['mean'])
        tcwv_std = torch.tensor(scaling['tcwv']['std'])

        p_levels = torch.ones((1, 1, 1, 1+len(self.hPa_levels), 1, 1))
        p_levels = p_levels * torch.tensor([0.] + self.hPa_levels).view(1, 1, 1, -1, 1, 1)
        p_levels = p_levels * 100.  # convert from hPa to Pa

        self.register_buffer('q_index', q_index, persistent=False)
        self.register_buffer('sp_index', sp_index, persistent=False)
        self.register_buffer('tcwv_index', tcwv_index, persistent=False)
        self.register_buffer('q_mean', q_mean, persistent=False)
        self.register_buffer('q_std', q_std, persistent=False)
        self.register_buffer('sp_mean', sp_mean, persistent=False)
        self.register_buffer('sp_std', sp_std, persistent=False)
        self.register_buffer('tcwv_mean', tcwv_mean, persistent=False)
        self.register_buffer('tcwv_std', tcwv_std, persistent=False)
        self.register_buffer('p_levels', p_levels, persistent=False)
        
    def forward(self, x):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        sp = torch.index_select(x, dim=3, index=self.sp_index)
        sp = sp * self.sp_std.view(1, 1, 1, -1, 1, 1) + self.sp_mean.view(1, 1, 1, -1, 1, 1)
        topography_mask = (self.p_levels[:,:,:,1:] > sp)

        q = torch.index_select(x, dim=3, index=self.q_index)
        q = q * self.q_std.view(1, 1, 1, -1, 1, 1) + self.q_mean.view(1, 1, 1, -1, 1, 1)
        q_toa = torch.zeros((q.shape[0], q.shape[1], q.shape[2], 1, q.shape[4], q.shape[5]), device=q.device, dtype=q.dtype)
        q = torch.cat((q_toa, q), dim=3)  # add q at TOA = 0 for computing integral via trapezoidal rule
        q = q * topography_mask

        delta_p = self.p_levels[:, :, :, 1:] - self.p_levels[:, :, :, :-1]
        tcwv = 0.5 * (q[:, :, :, 1:, :, :] + q[:, :, :, :-1, :, :]) * delta_p / self.g0
        tcwv = (tcwv - self.tcwv_mean.view(1, 1, 1, -1, 1, 1)) / self.tcwv_std.view(1, 1, 1, -1, 1, 1)

        return torch.cat((x, tcwv), dim=3)

class DeriveGeopotential(torch.nn.Module):
    def __init__(
        self,
        hPa_levels,
        output_channels,
        scaling,
        g0: float = 9.81,
    ):
        super().__init__()

        self.hPa_levels = hPa_levels
        self.output_channels = output_channels
        self.scaling = scaling
        self.g0 = g0

        for lev in self.hPa_levels:
            if f'z{lev}' in output_channels:
                raise ValueError(f'z{lev} already in output channels list.')
            if f'T{lev}' not in output_channels:
                raise ValueError(f'T{lev} not found in output channels list, cannot derive geopotential.')
            if f'q{lev}' not in output_channels:
                raise ValueError(f'q{lev} not found in output channels list, cannot derive geopotential.')

        if 'sp' not in output_channels:
            raise ValueError('sp not found in output channels list, cannot derive geopotential.')

        if not all(self.hPa_levels[i] < self.hPa_levels[i+1] for i in range(len(self.hPa_levels)-1)):
            raise ValueError('hPa_levels must be in increasing order (lowest to highest).')

        t_index = torch.tensor(
            [channels.index(f't{hPa_level}') for hPa_level in self.hPa_levels],
            dtype=torch.int
        )
        q_index = torch.tensor(
            [channels.index(f'q{hPa_level}') for hPa_level in self.hPa_levels],
            dtype=torch.int
        )
        sp_index = torch.tensor(output_channels.index('sp'), dtype=torch.int)

        t_mean = torch.tensor([scaling[f't{hPa_level}']['mean'] for hPa_level in self.hPa_levels])
        t_std = torch.tensor([scaling[f't{hPa_level}']['std'] for hPa_level in self.hPa_levels])
        q_mean = torch.tensor([scaling[f'q{hPa_level}']['mean'] for hPa_level in self.hPa_levels])
        q_std = torch.tensor([scaling[f'q{hPa_level}']['std'] for hPa_level in self.hPa_levels])
        sp_mean = torch.tensor(scaling['sp']['mean'])
        sp_std = torch.tensor(scaling['sp']['std'])

        p_levels = torch.ones((1, 1, 1, len(self.hPa_levels), 1, 1))
        p_levels = p_levels * torch.tensor(self.hPa_levels).view(1, 1, 1, -1, 1, 1)
        p_levels = p_levels * 100.  # convert from hPa to Pa

        self.register_buffer('t_index', t_index, persistent=False)
        self.register_buffer('q_index', q_index, persistent=False)
        self.register_buffer('sp_index', sp_index, persistent=False)
        self.register_buffer('t_mean', t_mean, persistent=False)
        self.register_buffer('t_std', t_std, persistent=False)
        self.register_buffer('q_mean', q_mean, persistent=False)
        self.register_buffer('q_std', q_std, persistent=False)
        self.register_buffer('sp_mean', sp_mean, persistent=False)
        self.register_buffer('sp_std', sp_std, persistent=False)
        self.register_buffer('p_levels', p_levels, persistent=False)
        
    def forward(self, x):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        sp = torch.index_select(x, dim=3, index=self.sp_index)
        sp = sp * self.sp_std.view(1, 1, 1, -1, 1, 1) + self.sp_mean.view(1, 1, 1, -1, 1, 1)
        topography_mask = (self.p_levels[:,:,:,1:] > sp)

        t = torch.index_select(x, dim=3, index=self.t_index)
        t = t * self.t_std.view(1, 1, 1, -1, 1, 1) + self.t_mean.view(1, 1, 1, -1, 1, 1)

        q = torch.index_select(x, dim=3, index=self.q_index)
        q = q * self.q_std.view(1, 1, 1, -1, 1, 1) + self.q_mean.view(1, 1, 1, -1, 1, 1)

        return