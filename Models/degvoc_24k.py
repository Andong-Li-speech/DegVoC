import torch 
import torch.nn as nn
import random

from .rnd_utils.basic_arch import *
from .rnd_utils.norm import *
from librosa.filters import mel as librosa_mel_fn


class Conv2FormerModule(nn.Module):
   def __init__(self,
                nband: int,
                input_channel: int,
                hidden_channel: int,
                f_kernel_size: int,
                t_kernel_size: int,
                mlp_ratio: int = 1,
                causal: bool = False,
                ):
      super(Conv2FormerModule, self).__init__()
      self.nband = nband
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.f_kernel_size = f_kernel_size
      self.t_kernel_size = t_kernel_size
      self.mlp_ratio = mlp_ratio
      self.causal = causal
      if self.causal:
         pad_ = nn.ConstantPad2d([t_kernel_size-1, 0, f_kernel_size//2, f_kernel_size//2], value=0.)
      else:
         pad_ = nn.ConstantPad2d([t_kernel_size//2, t_kernel_size//2, f_kernel_size//2, f_kernel_size//2], value=0.)
      # spatial attention
      self.attn = nn.Sequential(
         BandwiseC2LayerNorm(nband=nband, feature_dim=self.input_channel),
         nn.Conv2d(self.input_channel, self.hidden_channel, 1),
         nn.GELU(),
         pad_,
         nn.Conv2d(self.hidden_channel, self.hidden_channel, kernel_size=(f_kernel_size, t_kernel_size), groups=self.hidden_channel)
      )
      self.v = nn.Conv2d(self.input_channel, self.hidden_channel, 1)
      self.proj = nn.Conv2d(self.hidden_channel, self.input_channel, 1)

      # Feedforward
      self.fc1 = nn.Sequential(
         BandwiseC2LayerNorm(nband=nband, feature_dim=self.input_channel),
         nn.Conv2d(self.input_channel, self.input_channel * self.mlp_ratio, 1),
         nn.GELU()
      )
      if self.causal:
         pad_ = nn.ConstantPad2d([2, 0, 1, 1], value=0.)
      else:
         pad_ = nn.ConstantPad2d([1, 1, 1, 1], value=0.)
      self.dw_conv = nn.Sequential(
         pad_,
         nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel * self.mlp_ratio, 3, groups=self.input_channel * self.mlp_ratio),
         nn.GELU()
      )
      self.fc2 =  nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel, 1)
      
   def forward(self, x):
      """
      inpt: (B, C, nband, T)
      return: (B, C, nband, T)
      """
      # attn
      x_res = x
      x = self.attn(x) * self.v(x)
      x = self.proj(x)
      x = x_res + x
      # mlp
      x_res = x
      x = self.fc1(x)
      x = x + self.dw_conv(x)
      x = self.fc2(x)
      x = x_res + x
      return x


class DegVoC24k(nn.Module):
   def __init__(self, 
                h,
                ):
      super(DegVoC24k, self).__init__()
      self.h = h
      self.sampling_rate = h.sampling_rate
      self.num_mels = h.num_mels
      self.win_size = h.win_size 
      self.hop_size = h.hop_size
      self.n_fft = h.n_fft
      self.fmin = h.fmin
      self.fmax = h.fmax
      self.nstage = h.nstage
      self.input_channel = h.input_channel
      self.hidden_channel = h.hidden_channel
      self.f_kernel_size = h.f_kernel_size
      self.t_kernel_size = h.t_kernel_size
      self.mlp_ratio = h.mlp_ratio
      self.causal = h.causal
      self.use_shared_encoder = h.use_shared_encoder
      self.use_shared_decoder = h.use_shared_decoder
      self.decode_type = h.decode_type
      self.init_type = h.init_type
      self.phit_learnable = h.phit_learnable
      self.nb_num = h.nb_num
      self.use_even = h.use_even 
      self.eps = torch.finfo(torch.float32).eps

      # mel matrix
      mel = librosa_mel_fn(sr=self.sampling_rate,
                           n_fft=self.n_fft,
                           n_mels=self.num_mels,
                           fmin=self.fmin,
                           fmax=self.fmax,
                           )
      mel_basis = torch.from_numpy(mel)
      # inv_mel_basis = mel_basis.pinverse()
      if self.init_type == "pinv":
         inv_mel_basis = torch.linalg.pinv(mel_basis)
      elif self.init_type == "transpose":
         inv_mel_basis = mel_basis.T
      # Phi: (Fmel, F), PhiT: (F, Fmel)
      if self.phit_learnable:
         self.Phi = nn.Parameter(mel_basis, requires_grad=True)
         self.PhiT = nn.Parameter(inv_mel_basis, requires_grad=True)
      else:
         self.Phi = nn.Parameter(mel_basis, requires_grad=False)
         self.PhiT = nn.Parameter(inv_mel_basis, requires_grad=False)

      # null module
      if self.use_shared_encoder:
         # shared_22k also works herein
         if self.nb_num == 6:
            self.enc = SharedBandSplit_NB6_22k(sr=self.sampling_rate,
                                                win_size=self.win_size,
                                                hop_size=self.hop_size,
                                                n_fft=self.n_fft,
                                                feature_dim=self.input_channel,
                                                )
         elif self.nb_num == 12:
            self.enc = SharedBandSplit_NB12_22k(sr=self.sampling_rate,
                                                win_size=self.win_size,
                                                hop_size=self.hop_size,
                                                n_fft=self.n_fft,
                                                feature_dim=self.input_channel,
                                                )
         elif self.nb_num == 24:
            if self.use_even:
               self.enc = SharedBandSplit_NB24_even_22k(sr=self.sampling_rate,
                                                        win_size=self.win_size,
                                                        hop_size=self.hop_size,
                                                        n_fft=self.n_fft,
                                                        feature_dim=self.input_channel,
                                                        )
            else:
               self.enc = SharedBandSplit_NB24_22k(sr=self.sampling_rate,
                                                   win_size=self.win_size,
                                                   hop_size=self.hop_size,
                                                   n_fft=self.n_fft,
                                                   feature_dim=self.input_channel,
                                                   )
         elif self.nb_num == 48:
            self.enc = SharedBandSplit_NB48_22k(sr=self.sampling_rate,
                                                win_size=self.win_size,
                                                hop_size=self.hop_size,
                                                n_fft=self.n_fft,
                                                feature_dim=self.input_channel,
                                                )
         elif self.nb_num == 96:
            self.enc = SharedBandSplit_NB96_22k(sr=self.sampling_rate,
                                                win_size=self.win_size,
                                                hop_size=self.hop_size,
                                                n_fft=self.n_fft,
                                                feature_dim=self.input_channel,
                                                )
      else:
         self.enc = BandSplit_24k(sr=self.sampling_rate, 
                                  win_size=self.win_size,
                                  hop_size=self.hop_size,
                                  n_fft=self.n_fft,
                                  feature_dim=self.input_channel,
                                 )
      self.nband = self.enc.get_nband()
      if self.use_shared_decoder:
         if self.nb_num == 6:
            self.dec = SharedBandMerge_NB6_22k(sr=self.sampling_rate,
                                                     win_size=self.win_size,
                                                     hop_size=self.hop_size,
                                                     n_fft=self.n_fft,
                                                     feature_dim=self.input_channel,
                                                     decode_type=self.decode_type,
                                                    )
         elif self.nb_num == 12:
            self.dec = SharedBandMerge_NB12_22k(sr=self.sampling_rate,
                                                     win_size=self.win_size,
                                                     hop_size=self.hop_size,
                                                     n_fft=self.n_fft,
                                                     feature_dim=self.input_channel,
                                                     decode_type=self.decode_type,
                                                    )
         elif self.nb_num == 24:
            if self.use_even:
               self.dec = SharedBandMerge_NB24_even_22k(sr=self.sampling_rate,
                                                        win_size=self.win_size,
                                                        hop_size=self.hop_size,
                                                        n_fft=self.n_fft,
                                                        feature_dim=self.input_channel,
                                                        decode_type=self.decode_type,
                                                        )
            else:
               self.dec = SharedBandMerge_NB24_22k(sr=self.sampling_rate,
                                                   win_size=self.win_size,
                                                   hop_size=self.hop_size,
                                                   n_fft=self.n_fft,
                                                   feature_dim=self.input_channel,
                                                   decode_type=self.decode_type,
                                                   )
         elif self.nb_num == 48:
            self.dec = SharedBandMerge_NB48_22k(sr=self.sampling_rate,
                                                win_size=self.win_size,
                                                hop_size=self.hop_size,
                                                n_fft=self.n_fft,
                                                feature_dim=self.input_channel,
                                                decode_type=self.decode_type,
                                                )
         elif self.nb_num == 96:
            self.dec = SharedBandMerge_NB96_22k(sr=self.sampling_rate,
                                                win_size=self.win_size,
                                                hop_size=self.hop_size,
                                                n_fft=self.n_fft,
                                                feature_dim=self.input_channel,
                                                decode_type=self.decode_type,
                                                )
      else:
         self.dec = BandMerge_24k(sr=self.sampling_rate,
                                  win_size=self.win_size,
                                  hop_size=self.hop_size,
                                  n_fft=self.n_fft,
                                  feature_dim=self.input_channel,
                                  decode_type=self.decode_type,
                                 )
      module_list = []
      for _ in range(self.nstage): 
         module_list.append(
               Conv2FormerModule(nband=self.nband,
                                 input_channel=self.input_channel,
                                 hidden_channel=self.hidden_channel,
                                 f_kernel_size=self.f_kernel_size,
                                 t_kernel_size=self.t_kernel_size,
                                 mlp_ratio=self.mlp_ratio,
                                 causal=self.causal,
                                 )
         )

      self.module_list = nn.ModuleList(module_list)
      self.alpha = nn.Parameter(1e-4 * torch.ones([1, self.input_channel, self.nband, 1]))
      #
      self.apply(self._init_weights)

   def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

   def forward(self, mel):
      """
      mel: (B, F_mel, T)
      """
      # initialize
      init_mag = (self.PhiT @ torch.exp(mel)).abs().clamp_min_(1e-5)  # (B, F, T)
      init_real, init_imag = init_mag, torch.zeros_like(init_mag)
      init_spec = torch.stack([init_real, init_imag], dim=-1)  # (B, F, T, 2)
      x = self.enc(init_spec)
      x_res = x
      for i in range(self.nstage): 
         x = self.module_list[i](x)

      x_ = self.alpha * x_res + x

      if self.decode_type in ["mag+phase", "res+phase"]:
         res_mag, out_pha = self.dec(x_)
         if self.decode_type == "mag+phase":
            out_mag = res_mag
         else:
            out_mag = init_mag + res_mag
         out_spec = torch.stack([out_mag * torch.cos(out_pha), out_mag * torch.sin(out_pha)], dim=-1)
      elif self.decode_type in ["ri", "res_ri"]:
         res_real, res_imag = self.dec(x_)
         if self.decode_type == "ri":
            out_real, out_imag = res_real, res_imag
         else:
            out_real, out_imag = init_real + res_real, init_imag + res_imag
         out_spec = torch.stack([out_real, out_imag], dim=-1)

      logamp = torch.log(torch.norm(out_spec, dim=-1) + 1e-7)
      pha = torch.atan2(out_spec[..., -1], out_spec[..., 0])
      rea, imag = out_spec[..., 0], out_spec[..., -1]

      out_spec = torch.complex(rea, imag)
      out_wav = torch.istft(out_spec,
                            n_fft=self.n_fft,
                            hop_length=self.hop_size,
                            win_length=self.win_size,
                            window=torch.hann_window(self.win_size).to(mel.device))

      return logamp, pha, rea, imag, out_wav
