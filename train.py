import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import datetime
import itertools
import json
import math
import os
import random
import time
from typing import Any, Dict

import librosa as lib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import transformers
from pesq import pesq
from ptflops import get_model_complexity_info
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_libritts import (
    Dataset,
    amp_pha_specturm,
    get_dataset_filelist,
    mel_spectrogram,
    spectrogram,
)
from Models.degvoc_24k import DegVoC24k
from Models.models import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiResolutionMelLoss,
    OmniPhaseLoss,
    STFT_consistency_loss,
    Weighted_OmniPhaseLoss,
    amplitude_loss,
    discriminator_loss,
    feature_loss,
    generator_loss,
    phase_loss,
)
from utils import (
    AttrDict,
    build_env,
    load_checkpoint,
    plot_spectrogram,
    remove_older_checkpoint,
    save_checkpoint,
    scan_checkpoint,
)


MODEL_REGISTRY = {
    "DegVoC24k": DegVoC24k,
}


def set_random_seed(seed: int = 10, deterministic: bool = False, benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


def _to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    return obj


def expand_config(cfg: Dict[str, Any]):
    """
    Keep nested config for readability, while also exposing legacy flat fields so
    existing model/dataset code can continue to work without modification.
    """
    h = _to_attrdict(cfg)

    # defaults for optional fields used by the old training code
    h.setdefault("dist_backend", "nccl")
    h.setdefault("dist_url", "tcp://127.0.0.1:54321")
    h.setdefault("visualize_num", 10)
    h.setdefault("meloss", None)
    h.setdefault("phase_order", 1)
    h.setdefault("mrd_weight", 1.0)

    # expose nested sections as legacy flat fields
    if "data" in h:
        h.raw_wavfile_path = h.data.raw_wavfile_path
        h.input_training_wav_list = h.data.train_list
        h.input_validation_wav_list = h.data.valid_list

    if "train" in h:
        for key, value in h.train.items():
            h[key] = value

    if "model" in h:
        for key, value in h.model.items():
            h[key] = value

    if "audio" in h:
        for key, value in h.audio.items():
            h[key] = value

    if "loss" in h:
        h.use_omni_phase_loss = h.loss.use_omni_phase_loss
        h.use_mag_weighted = h.loss.use_mag_weighted
        h.mpd_reshapes = h.loss.mpd_reshapes
        h.mrd_resolutions = h.loss.mrd_resolutions
        h.mel_resolutions = h.loss.mel_resolutions
        h.loss_configs = h.loss.weights

    return h

def load_state_dict_flexible(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load checkpoint whether keys contain 'module.' prefix or not."""
    try:
        module.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    if all(k.startswith("module.") for k in state_dict.keys()):
        stripped = {k[len("module."):]: v for k, v in state_dict.items()}
        module.load_state_dict(stripped)
        return

    prefixed = {f"module.{k}": v for k, v in state_dict.items()}
    module.load_state_dict(prefixed)


def train(rank: int, a: argparse.Namespace, h: AttrDict):
    device_flag = rank == 0
    device = torch.device(f"cuda:{rank:d}")

    if a.num_gpus > 1:
        if device_flag:
            print("Using DDP for training.")
        dist.init_process_group(
            backend=h.dist_backend,
            init_method=h.dist_url,
            timeout=datetime.timedelta(seconds=3400),
            world_size=a.num_gpus,
            rank=rank,
        )

    h.batch_size_per_gpu = int(h.batch_size // a.num_gpus)
    set_random_seed(h.seed, deterministic=True)
    torch.cuda.set_device(rank)

    if device_flag:
        print(
            f"Let us use {a.num_gpus} GPUs in total, and for each gpu, "
            f"the batch size is {h.batch_size_per_gpu}."
        )

    if h.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name: {h.model_name}")
    generator = MODEL_REGISTRY[h.model_name](h)

    if device_flag:
        get_model_complexity_info(generator, (h.num_mels, h.sampling_rate // h.hop_size + 1))

    generator = generator.to(device)
    mpd = MultiPeriodDiscriminator(h.mpd_reshapes).to(device)
    mrd = MultiResolutionDiscriminator(resolutions=h.mrd_resolutions).to(device)
    ms_mel_loss = MultiResolutionMelLoss(
        resolutions=h.mel_resolutions,
        sampling_rate=h.sampling_rate,
    ).to(device)

    if device_flag:
        print("Using MultiPeriodDiscriminator")
        print("Using MultiResolutionDiscriminator.")

    omni_phase_loss = None
    if h.use_omni_phase_loss:
        if h.use_mag_weighted:
            omni_phase_loss = Weighted_OmniPhaseLoss(
                use_mag_weighted=h.use_mag_weighted,
                order=h.phase_order,
            ).to(device)
        else:
            omni_phase_loss = OmniPhaseLoss().to(device)

    os.makedirs(h.checkpoint_path, exist_ok=True)
    if device_flag:
        print("checkpoints directory:", h.checkpoint_path)

    cp_g = scan_checkpoint(h.checkpoint_path, "g_") if os.path.isdir(h.checkpoint_path) else None
    cp_do = scan_checkpoint(h.checkpoint_path, "do_") if os.path.isdir(h.checkpoint_path) else None

    steps = 0
    state_dict_do = None
    last_epoch = -1
    if cp_g is not None and cp_do is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        load_state_dict_flexible(generator, state_dict_g["generator"])
        load_state_dict_flexible(mpd, state_dict_do["mpd"])
        load_state_dict_flexible(mrd, state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]
        if device_flag:
            print(f"Resumed from step {steps}, epoch {last_epoch}.")

    if a.num_gpus > 1:
        generator = DDP(generator, device_ids=[rank], find_unused_parameters=True)
        mpd = DDP(mpd, device_ids=[rank], find_unused_parameters=True)
        mrd = DDP(mrd, device_ids=[rank], find_unused_parameters=True)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = transformers.get_cosine_schedule_with_warmup(
        optim_g,
        num_warmup_steps=0,
        num_training_steps=h.training_steps // 2,
    )
    scheduler_d = transformers.get_cosine_schedule_with_warmup(
        optim_d,
        num_warmup_steps=0,
        num_training_steps=h.training_steps // 2,
    )

    if state_dict_do is not None:
        scheduler_g.load_state_dict(state_dict_do["scheduler_g"])
        scheduler_d.load_state_dict(state_dict_do["scheduler_d"])

    training_filelist, validation_filelist = get_dataset_filelist(
        h.input_training_wav_list,
        h.input_validation_wav_list,
        h.raw_wavfile_path,
    )

    trainset = Dataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        n_cache_reuse=0,
        shuffle=True,
        device=device,
    )
    train_sampler = DistributedSampler(trainset) if a.num_gpus > 1 else None
    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False if a.num_gpus > 1 else True,
        sampler=train_sampler,
        batch_size=h.batch_size_per_gpu,
        pin_memory=True,
        drop_last=True,
    )

    validset = Dataset(
        validation_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        False,
        False,
        n_cache_reuse=0,
        device=device,
    )
    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )

    sw = SummaryWriter(os.path.join(h.checkpoint_path, "logs")) if device_flag else None

    generator.train()
    mpd.train()
    mrd.train()
    global_pesq_score = -math.inf

    for epoch in range(max(0, last_epoch), h.training_epochs):
        start = time.time()
        if device_flag:
            print(f"Epoch: {epoch + 1}")

        if a.num_gpus > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(tqdm(train_loader)):
            start_b = time.time()
            x, logamp, pha, rea, imag, y = map(
                lambda tensor: tensor.to(device, non_blocking=True), batch
            )

            logamp_g, pha_g, rea_g, imag_g, y_g = generator(x)

            if y_g.ndim == 3:
                y_g = y_g.squeeze(1)
            y_min = min(y_g.shape[-1], y.shape[-1])
            y_g, y = y_g[..., :y_min], y[..., :y_min]

            if i % 2 == 0:
                optim_d.zero_grad()

                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g.detach())
                loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

                y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g.detach())
                loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                l_d = (loss_disc_s * h.mrd_weight + loss_disc_f) * h.loss_configs["DiscriminatorLoss"]
                if torch.isnan(l_d).any():
                    print(f"Warning: Discriminator loss is NaN at step {steps}, batch {i}. Skipping this batch.")
                    continue

                l_d.backward()
                optim_d.step()
            else:
                optim_g.zero_grad()

                l_a = amplitude_loss(logamp, logamp_g)

                if not h.use_omni_phase_loss:
                    l_p = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1])
                else:
                    if h.use_mag_weighted:
                        l_p = omni_phase_loss(pha, pha_g, torch.exp(logamp))
                    else:
                        l_p = omni_phase_loss(pha, pha_g)

                _, _, rea_g_final, imag_g_final = amp_pha_specturm(y_g, h.n_fft, h.hop_size, h.win_size)
                l_c = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final)

                l_r = F.l1_loss(rea, rea_g)
                l_i = F.l1_loss(imag, imag_g)
                l_ri = l_r + l_i

                _, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_g)
                _, y_ds_g, fmap_s_r, fmap_s_g = mrd(y, y_g)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, _ = generator_loss(y_df_g)
                loss_gen_s, _ = generator_loss(y_ds_g)
                l_gan_g = loss_gen_s * h.mrd_weight + loss_gen_f
                l_fm = loss_fm_s * h.mrd_weight + loss_fm_f
                l_mel = ms_mel_loss(y=y, y_hat=y_g)

                l_g = (
                    h.loss_configs["AmplitudeLoss"] * l_a
                    + h.loss_configs["PhaseLoss"] * l_p
                    + h.loss_configs["STFTConsistencyLoss"] * l_c
                    + h.loss_configs["RILoss"] * l_ri
                    + h.loss_configs["GeneratorLoss"] * l_gan_g
                    + h.loss_configs["FeatureMatchingLoss"] * l_fm
                    + h.loss_configs["MelSpecReconstructLoss"] * l_mel
                )

                if torch.isnan(l_g).any():
                    print(f"Warning: Generator loss is NaN at step {steps}, batch {i}. Skipping this batch.")
                    continue

                l_g.backward()
                if h.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), h.clip_grad_norm)
                optim_g.step()

            if steps % h.stdout_interval == 0 and steps != 0:
                with torch.no_grad():
                    a_error = amplitude_loss(logamp, logamp_g).item()
                    if not h.use_omni_phase_loss:
                        phase_error = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1]).item()
                    else:
                        if h.use_mag_weighted:
                            phase_error = omni_phase_loss(pha, pha_g, torch.exp(logamp)).item()
                        else:
                            phase_error = omni_phase_loss(pha, pha_g).item()

                    c_error = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final).item()
                    r_error = F.l1_loss(rea, rea_g).item()
                    i_error = F.l1_loss(imag, imag_g).item()
                    mel_error = ms_mel_loss(y=y, y_hat=y_g).item()

                if device_flag:
                    print(
                        "Steps: {:d}, Gen Loss Total: {:4.3f}, Amplitude Loss: {:4.3f}, "
                        "Phase Loss: {:4.3f}, STFT Consistency Loss: {:4.3f}, Real Part Loss: {:4.3f}, "
                        "Imaginary Part Loss: {:4.3f}, Mel Spectrogram Loss: {:4.3f}, s/b: {:4.3f}".format(
                            steps,
                            l_g if i % 2 == 1 else float("nan"),
                            a_error,
                            phase_error,
                            c_error,
                            r_error,
                            i_error,
                            mel_error,
                            time.time() - start_b,
                        )
                    )

            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = f"{h.checkpoint_path}/g_{steps:08d}"
                save_checkpoint(
                    checkpoint_path,
                    {"generator": (generator.module if a.num_gpus > 1 else generator).state_dict()},
                )
                remove_older_checkpoint(checkpoint_path, pre="g", max_to_keep=h.max_to_keep)

                checkpoint_path = f"{h.checkpoint_path}/do_{steps:08d}"
                save_checkpoint(
                    checkpoint_path,
                    {
                        "mpd": (mpd.module if a.num_gpus > 1 else mpd).state_dict(),
                        "mrd": (mrd.module if a.num_gpus > 1 else mrd).state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "scheduler_d": scheduler_d.state_dict(),
                        "scheduler_g": scheduler_g.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    },
                )
                remove_older_checkpoint(checkpoint_path, pre="d", max_to_keep=h.max_to_keep)

            if device_flag and sw is not None and steps % h.summary_interval == 0 and steps != 0 and i % 2 == 1:
                sw.add_scalar("Training/Generator_Total_Loss", l_g.item(), steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", l_mel.item(), steps)

            if device_flag and steps % h.validation_interval == 0 and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_a_err_tot = 0.0
                val_p_err_tot = 0.0
                val_c_err_tot = 0.0
                val_r_err_tot = 0.0
                val_i_err_tot = 0.0
                val_mel_err_tot = 0.0
                pesq_tot = 0.0

                with torch.no_grad():
                    for j, batch in enumerate(tqdm(validation_loader)):
                        x, logamp, pha, rea, imag, y = map(
                            lambda tensor: tensor.to(device, non_blocking=True), batch
                        )
                        outputs = generator.module(x) if hasattr(generator, "module") else generator(x)
                        logamp_g, pha_g, rea_g, imag_g, y_g = outputs

                        if y_g.ndim == 3:
                            y_g = y_g.squeeze(1)
                        y_min = min(y_g.shape[-1], y.shape[-1])
                        y_g, y = y_g[..., :y_min], y[..., :y_min]

                        _, _, rea_g_final, imag_g_final = amp_pha_specturm(y_g, h.n_fft, h.hop_size, h.win_size)
                        val_a_err_tot += amplitude_loss(logamp, logamp_g).item()

                        if not h.use_omni_phase_loss:
                            val_p_err = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1])
                        else:
                            if h.use_mag_weighted:
                                val_p_err = omni_phase_loss(pha, pha_g, torch.exp(logamp))
                            else:
                                val_p_err = omni_phase_loss(pha, pha_g)
                        val_p_err_tot += val_p_err.item()
                        val_c_err_tot += STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final).item()
                        val_r_err_tot += F.l1_loss(rea, rea_g).item()
                        val_i_err_tot += F.l1_loss(imag, imag_g).item()
                        val_mel_err_tot += ms_mel_loss(y=y, y_hat=y_g).item()

                        y_g_np = y_g.cpu().squeeze().numpy()
                        y_np = y.cpu().squeeze().numpy()
                        if h.sampling_rate != 16000:
                            y_g_np = lib.core.resample(y_g_np, orig_sr=h.sampling_rate, target_sr=16000)
                            y_np = lib.core.resample(y_np, orig_sr=h.sampling_rate, target_sr=16000)
                        pesq_tot += pesq(16000, y_np, y_g_np, mode="wb")

                        if sw is not None and j <= h.visualize_num:
                            sw.add_audio(f"gt/y_{j}", y[0], steps, h.sampling_rate)
                            y_spec = spectrogram(
                                y,
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax,
                            )
                            sw.add_figure(
                                f"gt/y_spec_{j}",
                                plot_spectrogram(y_spec.squeeze(0).cpu()),
                                steps,
                            )
                            sw.add_audio(f"generated/y_g_{j}", y_g[0], steps, h.sampling_rate)
                            y_g_spec = spectrogram(
                                y_g,
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax,
                            )
                            sw.add_figure(
                                f"generated/y_g_spec_{j}",
                                plot_spectrogram(y_g_spec.squeeze(0).cpu().numpy()),
                                steps,
                            )

                    denom = j + 1
                    val_a_err = val_a_err_tot / denom
                    val_p_err = val_p_err_tot / denom
                    val_c_err = val_c_err_tot / denom
                    val_r_err = val_r_err_tot / denom
                    val_i_err = val_i_err_tot / denom
                    val_mel_err = val_mel_err_tot / denom
                    val_pesq_score = pesq_tot / denom

                    if sw is not None:
                        sw.add_scalar("Validation/Amplitude_Loss", val_a_err, steps)
                        sw.add_scalar("Validation/Phase_Loss", val_p_err, steps)
                        sw.add_scalar("Validation/STFT_Consistency_Loss", val_c_err, steps)
                        sw.add_scalar("Validation/Real_Part_Loss", val_r_err, steps)
                        sw.add_scalar("Validation/Imaginary_Part_Loss", val_i_err, steps)
                        sw.add_scalar("Validation/Mel_Spectrogram_loss", val_mel_err, steps)
                        sw.add_scalar("Validation/PESQ_score", val_pesq_score, steps)

                    if h.save_best and val_pesq_score >= global_pesq_score:
                        ckp_best_path = f"{h.checkpoint_path}/best_g"
                        save_checkpoint(
                            ckp_best_path,
                            {"generator": (generator.module if a.num_gpus > 1 else generator).state_dict()},
                        )
                        global_pesq_score = val_pesq_score

                generator.train()

            steps += 1
            if steps == h.training_steps + 1:
                if sw is not None:
                    sw.close()
                if a.num_gpus > 1:
                    dist.destroy_process_group()
                return

            if i % 2 == 0:
                scheduler_d.step()
            else:
                scheduler_g.step()

        if device_flag:
            print(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")

    if sw is not None:
        sw.close()
    if a.num_gpus > 1:
        dist.destroy_process_group()


def main():
    print("Initializing Training Process...")
    parser = argparse.ArgumentParser("Vocoder configs.")
    parser.add_argument(
        "--cfg_filename",
        type=str,
        required=True,
        help="Json for configurations.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="",
        help="GPU id(s) to use, e.g. '0' or '0,1'. Leave empty to use all visible GPUs.",
    )
    args = parser.parse_args()

    if args.gpu != "":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config_file = args.cfg_filename
    config_filename = os.path.basename(config_file)

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    h = expand_config(cfg)
    build_env(config_file, config_filename, h.checkpoint_path)

    args.num_gpus = torch.cuda.device_count()
    print(f"Visible GPUs: {args.gpu if args.gpu != '' else 'all'}")
    print(f"Detected num_gpus: {args.num_gpus}")

    if args.num_gpus < 1:
        raise RuntimeError("No GPU detected. Training requires at least one CUDA device.")

    if args.num_gpus > 1:
        mp.spawn(train, nprocs=args.num_gpus, args=(args, h))
    else:
        train(0, args, h)


if __name__ == "__main__":
    main()
