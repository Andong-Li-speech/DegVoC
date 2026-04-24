
import os, sys
from typing import List
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from encodec import EncodecModel
import dac
from audiotools import AudioSignal
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from WavTokenizer_Project.encoder.utils import convert_audio
from WavTokenizer_Project.decoder.pretrained import WavTokenizer


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_window = {}
inv_mel_window = {}


def param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device):
    return f"{sampling_rate}-{n_fft}-{num_mels}-{fmin}-{fmax}-{win_size}-{device}"


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=True,
):
    global mel_window
    device = y.device
    ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if ps in mel_window:
        mel_basis, hann_window = mel_window[ps]
        # print(mel_basis, hann_window)
        # mel_basis, hann_window = mel_basis.to(y.device), hann_window.to(y.device)
    else:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis = torch.from_numpy(mel).float().to(device)
        hann_window = torch.hann_window(win_size).to(device)
        mel_window[ps] = (mel_basis.clone(), hann_window.clone())

    spec = torch.stft(
        y.to(device),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window.to(device),
        center=True,
        return_complex=True,
    )

    spec = mel_basis.to(device) @ spec.abs()
    spec = spectral_normalize_torch(spec)

    return spec  # [batch_size,n_fft/2+1,frames]


def inverse_mel(
    mel,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
):
    global inv_mel_window, mel_window
    device = mel.device
    ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if ps in inv_mel_window:
        inv_basis = inv_mel_window[ps]
        mel_basis = (mel_window[ps])[0]
    else:
        if ps in mel_window:
            mel_basis, _ = mel_window[ps]
        else:
            mel_np = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            mel_basis = torch.from_numpy(mel_np).float().to(device)
            hann_window = torch.hann_window(win_size).to(device)
            mel_window[ps] = (mel_basis.clone(), hann_window.clone())
        inv_basis = mel_basis.pinverse()
        inv_mel_window[ps] = inv_basis.clone()
    
    return mel_basis.to(device), inv_basis.to(device), inv_basis.to(device) @ spectral_de_normalize_torch(mel.to(device))
        


class MelSpectrogramFeatures(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=1024, win_size=1024, hop_size=256, num_mels=100, fmin=0, fmax=12000, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels

    def mel_forward(self, audio):
        """
        audio: (B, L)
        return: (B, Fmel, T), log-format
        """
        inpt_mel = mel_spectrogram(audio, 
                                    n_fft=self.n_fft,
                                    num_mels=self.num_mels,
                                    sampling_rate=self.sample_rate,
                                    hop_size=self.hop_size,
                                    win_size=self.win_size,
                                    fmin=self.fmin,
                                    fmax=self.fmax,
                                    )
        return inpt_mel
    
    def inverse_mel_forward(self, inpt):
        """
        inpt: (B, Fmel, T)
        return: (B, F, T), non-log format
        """
        inpt_x = (inverse_mel(inpt, 
                            n_fft=self.n_fft,
                            num_mels=self.num_mels,
                            sampling_rate=self.sample_rate,
                            hop_size=self.hop_size,
                            win_size=self.win_size,
                            fmin=self.fmin,
                            fmax=self.fmax,
                            ))[-1].abs().clamp_min_(1e-6)
        return inpt_x


class DACFeatures(nn.Module):
    def __init__(self,
                 sample_rate: int,
                 pretrained_path: str = "",
                 bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.pretrained_path = pretrained_path
        self.bandwidths = bandwidths 
        try:
            self.model = dac.DAC.load(pretrained_path)
        except:
            self.model = dac.DAC  # randomly initialization
        for param in self.model.parameters():
            param.requires_grad = False
        self.frame_rate = sample_rate / np.prod(self.model.encoder_rates)
        self.kbps = self.frame_rate * np.log2(self.model.codebook_size) / 1000
    
    @torch.no_grad()
    def forward(self, audio: torch.Tensor, **kwargs):
        """
        audio: (B, L)
        return: (B, C, T)
        """
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")
        self.model.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        num_quantizers = int(self.bandwidths[bandwidth_id] / self.kbps)
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)  # (B, 1, L)
        x = self.model.preprocess(audio, self.sample_rate)
        features = self.model.encode(x, n_quantizers=num_quantizers)[0]
        return features


class EncodecFeatures(nn.Module):
    def __init__(self, 
                  encodec_model: str = "encodec_24khz",
                  bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
                  train_codebooks: bool = False,
                  ):
        super().__init__()
        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
                self.encodec.frame_rate, bandwidth=max(bandwidths)
        )
        codebook_weights = torch.cat([vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]], dim=0)
        self.codebook_weights = torch.nn.Parameter(codebook_weights, requires_grad=train_codebooks)
        self.bandwidths = bandwidths

    @torch.no_grad()
    def get_encodec_codes(self, audio):
        """
        audio: (B, L)
        return: (N, B, T)
        """
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, **kwargs):
        """
        audio: (B, L)
        return: (B, C, T)
        """
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")
        self.encodec.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])
        codes = self.get_encodec_codes(audio)
        # Instead of summing in the loop, it stores subsequent VQ dictionaries in a single `self.codebook_weights`
        # with offsets given by the number of bins, and finally summed in a vectorized operation.
        offsets = torch.arange(
                0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins, device=audio.device
            )
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)  # (N,B,T)->(N,B,T,C)->(B,T,C)
        return features.transpose(1, 2)


class WavTokenizerFeatures(nn.Module):
    def __init__(self,
                 sampling_rate: int,
                 config_path: str,
                 model_path: str,
                 ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.config_path = config_path
        self.model_path = model_path
        #
        self.model = WavTokenizer.from_pretrained0802(self.config_path, self.model_path)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, audio: torch.Tensor, **kwargs):
        """
        audio: (B, L)
        return: (B, C, T)
        """
        try:
            audio = convert_audio(audio, self.sampling_rate, 24000, 1)
        except:
            pass
        bandwidth_id = torch.tensor([0])
        features, discrete_code= self.model.encode_infer(audio, bandwidth_id=bandwidth_id)
        return features
    
    @torch.no_grad()
    def decoder(self, feature: torch.Tensor):
        """
        audio: (B, C, T)
        return: (B, L)
        """
        est_wav = self.model.decode(feature)
        return est_wav
    # @torch.no_grad()
    # def forward(self, audio: torch.Tensor, **kwargs):
    #     """
    #     audio: (B, L)
    #     return: (B, C, T)
    #     """
    #     try:
    #         audio = convert_audio(audio, self.sampling_rate, 24000, 1)
    #     except:
    #         pass
    #     bandwidth_id = torch.tensor([0])
    #     features, discrete_code= self.model.encode_infer(audio, bandwidth_id=bandwidth_id)
    #     est_wav = self.model.decode(features, bandwidth_id=bandwidth_id)
    #     return est_wav
    

def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))
