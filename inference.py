from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import librosa as lib
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from utils import AttrDict
from dataset_libritts import mel_spectrogram
from Models.degvoc_24k import DegVoC24k


MODEL_REGISTRY = {
    "DegVoC24k": DegVoC24k,
}


def load_checkpoint(filepath: str, device: torch.device):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    print(f"Loading checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    print("Checkpoint loaded.")
    return checkpoint

def to_attr_dict(obj):
    if isinstance(obj, dict):
        return AttrDict({k: to_attr_dict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_attr_dict(v) for v in obj]
    else:
        return obj

def load_config(cfg_filename: str) -> AttrDict:
    if not os.path.isfile(cfg_filename):
        raise FileNotFoundError(f"Config file not found: {cfg_filename}")
    with open(cfg_filename, "r", encoding="utf-8") as f:
        config = json.load(f)
    return to_attr_dict(config)

def flatten_config_for_legacy_code(h: AttrDict):
    """
    Expand nested config fields into the flat attribute names expected by
    the current model / dataset code, while preserving nested access.
    """
    if "data" in h:
        print(f"{h.data}")
        h.raw_wavfile_path = h.data.raw_wavfile_path
        h.input_training_wav_list = h.data.train_list
        h.input_validation_wav_list = h.data.valid_list

    if "train" in h:
        h.batch_size = h.train.batch_size
        h.learning_rate = h.train.learning_rate
        h.adam_b1 = h.train.adam_b1
        h.adam_b2 = h.train.adam_b2
        h.training_steps = h.train.training_steps
        h.training_epochs = h.train.training_epochs
        h.stdout_interval = h.train.stdout_interval
        h.checkpoint_interval = h.train.checkpoint_interval
        h.summary_interval = h.train.summary_interval
        h.validation_interval = h.train.validation_interval
        h.clip_grad_norm = h.train.clip_grad_norm
        h.seed = h.train.seed
        h.num_workers = h.train.num_workers
        h.checkpoint_path = h.train.checkpoint_path
        h.max_to_keep = h.train.max_to_keep
        h.save_best = h.train.save_best

    if "model" in h:
        h.nstage = h.model.nstage
        h.input_channel = h.model.input_channel
        h.hidden_channel = h.model.hidden_channel
        h.nb_num = h.model.nb_num
        h.use_even = h.model.use_even
        h.f_kernel_size = h.model.f_kernel_size
        h.t_kernel_size = h.model.t_kernel_size
        h.mlp_ratio = h.model.mlp_ratio
        h.causal = h.model.causal
        h.use_shared_encoder = h.model.use_shared_encoder
        h.use_shared_decoder = h.model.use_shared_decoder
        h.decode_type = h.model.decode_type
        h.init_type = h.model.init_type
        h.phit_learnable = h.model.phit_learnable

    if "audio" in h:
        h.sampling_rate = h.audio.sampling_rate
        h.segment_size = h.audio.segment_size
        h.num_mels = h.audio.num_mels
        h.n_fft = h.audio.n_fft
        h.hop_size = h.audio.hop_size
        h.win_size = h.audio.win_size
        h.fmin = h.audio.fmin
        h.fmax = h.audio.fmax

    if "loss" in h:
        h.use_omni_phase_loss = h.loss.use_omni_phase_loss
        h.use_mag_weighted = h.loss.use_mag_weighted
        h.mpd_reshapes = h.loss.mpd_reshapes
        h.mrd_resolutions = h.loss.mrd_resolutions
        h.mel_resolutions = h.loss.mel_resolutions
        h.loss_configs = h.loss.weights

    return h


def get_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_arg)


def build_model(h: AttrDict, device: torch.device):
    if h.model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_name: {h.model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    model = MODEL_REGISTRY[h.model_name](h).to(device)
    return model

def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned

def load_generator_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = load_checkpoint(checkpoint_path, device)
    if "generator" in ckpt:
        state_dict = ckpt["generator"]
    else:
        state_dict = ckpt
    state_dict = clean_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading checkpoint: {unexpected}")

def get_mel(wav_tensor: torch.Tensor, h: AttrDict):
    return mel_spectrogram(
        wav_tensor,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax,
    )

def ensure_mono(audio: np.ndarray):
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)

def normalize_audio(audio: np.ndarray, eps: float = 1e-9):
    peak = np.max(np.abs(audio))
    if peak < eps:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)

def resolve_wav_list(input_path: str, raw_wavfile_path: str, dataset_type: str):
    input_path = str(input_path)
    suffix = Path(input_path).suffix.lower()
    dataset_type = dataset_type.lower()

    if suffix in [".txt", ".scp"]:
        filelist = []
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if dataset_type == "libritts":
                cur_filename = line.split("|")[0] + ".wav"
                filelist.append(os.path.join(raw_wavfile_path, cur_filename))
            elif dataset_type == "ljspeech":
                cur_filename = line.split("/")[1].split("|")[0]
                filelist.append(os.path.join(raw_wavfile_path, cur_filename))
            else:
                if raw_wavfile_path:
                    filelist.append(os.path.join(raw_wavfile_path, line))
                else:
                    filelist.append(line)
        return filelist

    if os.path.isdir(input_path):
        valid_suffixes = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        return [
            str(p) for p in sorted(Path(input_path).iterdir())
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ]

    if os.path.isfile(input_path):
        return [input_path]

    raise FileNotFoundError(f"Input wav path not found: {input_path}")


def resolve_mel_list(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return [
            str(p) for p in sorted(Path(input_path).iterdir())
            if p.is_file() and p.suffix.lower() == ".npy"
        ]
    if os.path.isfile(input_path) and Path(input_path).suffix.lower() == ".npy":
        return [input_path]
    raise FileNotFoundError(f"Input mel path not found or not a .npy file/dir: {input_path}")


def load_audio_as_mel(wav_path: str, h: AttrDict, device: torch.device):
    wav, sr = sf.read(wav_path)
    wav = ensure_mono(wav).astype(np.float32)
    if sr != h.sampling_rate:
        wav = lib.resample(wav, orig_sr=sr, target_sr=h.sampling_rate)
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)
    mel = get_mel(wav_tensor, h)
    return mel, wav_tensor.shape[-1]


def load_mel_npy(mel_path: str, device: torch.device):
    mel = np.load(mel_path)
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim == 2:
        mel = mel[None, ...]
    elif mel.ndim != 3:
        raise ValueError(f"Expected mel ndim 2 or 3, but got shape {mel.shape} from {mel_path}")
    return torch.from_numpy(mel).to(device)


def forward_generator(generator: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = generator(x)
    if isinstance(output, torch.Tensor):
        y = output
    elif isinstance(output, (list, tuple)):
        y = output[-1]
    else:
        raise TypeError(f"Unsupported generator output type: {type(output)}")

    if y.ndim == 3 and y.shape[1] == 1:
        y = y.squeeze(1)
    return y


def save_audio(audio: np.ndarray, save_path: str, sampling_rate: int, normalize: bool = False):
    if normalize:
        audio = 0.5 * normalize_audio(audio)
    sf.write(save_path, audio, sampling_rate, subtype="PCM_16")


def run_inference(args, h: AttrDict):
    device = get_device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.test_output_dir, exist_ok=True)

    generator = build_model(h, device)
    load_generator_weights(generator, args.checkpoint_file_load, device)
    generator.eval()

    try:
        generator.remove_weight_norm()
    except Exception:
        pass

    if args.use_mel_format:
        filelist = resolve_mel_list(args.test_input_mels_dir)
    else:
        filelist = resolve_wav_list(args.test_input_wavs_dir, h.raw_wavfile_path, h.dataset_type)

    if len(filelist) == 0:
        raise RuntimeError("No input files found for inference.")

    print(f"Found {len(filelist)} input files.")

    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for filepath in tqdm(filelist, desc="Inferencing"):
            filepath = str(filepath)
            stem = Path(filepath).stem
            save_path = os.path.join(args.test_output_dir, stem + ".wav")

            if args.use_mel_format:
                x = load_mel_npy(filepath, device)
                # Keep compatibility with the current code path: mel is assumed to be log-mel.
                x = torch.log(torch.clamp(torch.exp(x), min=1e-5))
            else:
                x, _ = load_audio_as_mel(filepath, h, device)

            y = forward_generator(generator, x)
            audio = y.squeeze(0).detach().cpu().numpy().astype(np.float32)

            total_samples += len(audio)
            save_audio(
                audio,
                save_path,
                h.sampling_rate,
                normalize=args.normalize_output or args.use_mel_format,
            )

    elapsed = time.time() - start_time
    total_seconds = total_samples / float(h.sampling_rate)
    throughput = total_seconds / max(elapsed, 1e-8)

    print(f"Elapsed time: {elapsed:.3f} s")
    print(f"Generated audio duration: {total_seconds:.3f} s")
    print(f"Throughput (audio_seconds / wall_seconds): {throughput:.3f}")


def parse_args():
    parser = argparse.ArgumentParser("DegVoC inference")
    parser.add_argument("--cfg_filename", type=str, required=True, help="Path to config json.")
    parser.add_argument(
        "--test_input_wavs_dir",
        type=str,
        default="",
        help="Input wav path. Can be a wav file, a directory, or a .txt/.scp filelist.",
    )
    parser.add_argument(
        "--test_input_mels_dir",
        type=str,
        default="",
        help="Input mel path. Can be a .npy file or a directory of .npy files.",
    )
    parser.add_argument(
        "--use_mel_format",
        action="store_true",
        help="Use mel .npy inputs instead of wav inputs.",
    )
    parser.add_argument("--test_output_dir", type=str, required=True, help="Directory to save generated wavs.")
    parser.add_argument("--checkpoint_file_load", type=str, required=True, help="Path to generator checkpoint.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--normalize_output",
        action="store_true",
        help="Peak-normalize output audio before saving.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    h = load_config(args.cfg_filename)
    h = flatten_config_for_legacy_code(h)

    if args.use_mel_format:
        if not args.test_input_mels_dir:
            raise ValueError("--use_mel_format is set, but --test_input_mels_dir is empty.")
    else:
        if not args.test_input_wavs_dir:
            raise ValueError("--use_mel_format is not set, but --test_input_wavs_dir is empty.")

    run_inference(args, h)


if __name__ == "__main__":
    main()
