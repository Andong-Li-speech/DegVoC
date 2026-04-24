<div align="center">

# 🎧 DegVoC: Revisiting Neural Vocoder from a Degradation Perspective

**Andong Li, Tong Lei, Lingling Dai, Kai Li, Rilin Chen, Meng Yu, Xiaodong Li, Dong Yu, Chengshi Zheng**

**🏆 Accepted by AAAI 2026** · [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/40416) · [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/40416/44507) · [DOI](https://doi.org/10.1609/aaai.v40i37.40416)

<!-- Optional badges: replace the Hugging Face path after release -->
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/AndongLi/DegVoC)

</div>

---

## ✨ Overview

DegVoC revisits neural vocoding from a **degradation perspective** rather than the conventional “generate waveform from scratch” viewpoint. Instead of treating the Mel-spectrum as a generic acoustic condition, DegVoC models it as a **degraded observation** of the target spectrum and solves vocoding through a **two-step restoration pipeline**:

1. **Initialization solver**: recover a coarse spectral structure from the Mel domain via a lightweight linear inverse transformation.
2. **Deep prior solver**: refine the spectrum with a subband-aware neural network that captures heterogeneous time-frequency priors.

This design leads to a strong balance between **generation quality**, **model size**, and **inference efficiency**.

---

## 🔥 Core Ideas

- **Reformulating vocoding as restoration.**  
  DegVoC interprets the Mel-spectrum as the result of **linear magnitude compression + phase removal**, and casts neural vocoding as an inverse problem rather than conditional generation from Gaussian noise.

- **Two-step optimization-inspired pipeline.**  
  The target spectrum is recovered with:
  - an **initialization solver** for coarse spectral reconstruction, and
  - a **deep prior solver** for high-fidelity refinement.

- **Degradation-aware spectral initialization.**  
  DegVoC exploits the Mel degradation prior through matrix-based initialization strategies such as **transpose**, **pseudo-inverse**, and **learnable linear initialization**.

- **Subband-aware prior learning.**  
  Instead of modeling the whole spectrum uniformly, DegVoC adopts an **uneven hierarchical subband division/merge strategy** to better fit the heterogeneous distribution of low-, mid-, and high-frequency regions.

- **Large-kernel convolutional attention.**  
  The proposed **LKCAM** models both **inter-frame** and **inter-band** dependencies using large-kernel depthwise convolutions, enabling efficient contextual modeling without heavy self-attention overhead.

- **Lightweight yet strong.**  
  With only **3.89M parameters**, DegVoC achieves state-of-the-art or highly competitive performance while keeping inference efficient.

---

## 📈 Why DegVoC Matters

- **Optimization-inspired** rather than generation-from-scratch.
- **One-pass GAN inference** instead of iterative diffusion / flow sampling.
- **Subband-aware T-F modeling** with strong efficiency-quality trade-off.
- **Competitive objective and subjective performance** on both in-domain and out-of-distribution benchmarks.

---

## 🗂️ Project Structure

```text
DegVoC/
├── cfgs/
│   └── degvoc_libritts_config.json
├── ckpts/
├── DatasetsScp/
│   └── LibriTTS/
├── Models/
│   ├── degvoc_24k.py
│   ├── models.py
│   └── rnd_utils/
├── dataset_libritts.py
├── train.py
├── inference.py
├── train.sh
├── infer_wav.sh
└── utils.py
```

---

## 🛠️ Installation

We recommend Python **3.9+** and a CUDA-enabled PyTorch environment.

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

### 1. Prepare the dataset

The default configuration is built on LibriTTS at 24 kHz. Please update the dataset paths in:

```text
cfgs/degvoc_libritts_config.json
```

In particular:

- `data.raw_wavfile_path`
- `data.train_list`
- `data.valid_list`

### 2. Launch training

A stable way is to specify visible GPUs **before** launching Python:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --cfg_filename cfgs/degvoc_libritts_config.json\
  --gpu 0,1
```

Single-GPU training is also supported:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_filename cfgs/degvoc_libritts_config.json \
  --gpu 0
```

---

## 🎙️ Inference

### Inference from input wavs

```bash
python inference.py \
  --cfg_filename ./cfgs/degvoc_libritts_config.json \
  --test_input_wavs_dir /path/to/test_wavs \
  --test_output_dir ./generated \
  --checkpoint_file_load ./ckpts/g_02000000_n8.pt \
  --device auto
```

### Inference from precomputed Mel features

```bash
python inference.py \
  --cfg_filename ./cfgs/degvoc_libritts_config.json \
  --test_input_mels_dir /path/to/test_mels \
  --use_mel_format \
  --test_output_dir ./generated \
  --checkpoint_file_load ./ckpts/g_02000000_n8.pt \
  --device auto
```

---

## 🤗 Hugging Face

| Model Name                   | Dataset | Sample Rate | NMel | Training Steps |
|------------------------------|---------|-------------|------|----------------|
| 🤗 [DegVoC-LibriTTS](https://huggingface.co/AndongLi/DegVoC/blob/main/g_02000000_n8.pt) | LibriTTS | 24k Hz | 100 | 1M Generator + 1M Discriminator |

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026degvoc,
  title     = {DegVoC: Revisiting Neural Vocoder from a Degradation Perspective},
  author    = {Li, Andong and Lei, Tong and Dai, Lingling and Li, Kai and Chen, Rilin and Yu, Meng and Li, Xiaodong and Yu, Dong and Zheng, Chengshi},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {37},
  pages     = {31510--31518},
  year      = {2026},
  doi       = {10.1609/aaai.v40i37.40416},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/40416}
}
```

---

## 🙏 Acknowledgement

This work was supported by the National Natural Science Foundation of China (NSFC) under Grant 62501588.
