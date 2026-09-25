# FL_paper — Adap-CTA: Federated Learning Privacy Codebase

Research codebase for a federated learning (FL) privacy paper, built on
[PFLlib](https://github.com/TsingZ0/PFLlib). It implements the proposed
**Adap-CTA** protocol, a set of **gradient-inversion attacks** that try to
reconstruct client training data from the model updates the server sees, and
privacy-defense baselines (differential privacy, homomorphic encryption) used
to evaluate how well those attacks can be resisted.

**Threat model.** An honest-but-curious server observes exactly what a client
uploads (model update / gradient transcript). The attacks reconstruct images
from that transcript alone; ground-truth client data is used only for
evaluation metrics (PSNR / MSE / SSIM), never as attack input.

## Repository layout

```
dataset/                      # dataset preparation scripts ONLY (raw data is NOT included)
├── generate_cifar10.py       # download/split CIFAR-10
└── utils/                    # dataset_utils.py (non-IID partitioning), language_utils.py (Shakespeare)

system/system/                # main code (PFLlib-style layout, run everything from here)
├── main.py                   # FL training entry point (-algo selects the protocol)
├── main_collect.py           # batch experiment runner
├── export_attack_bundle.py   # exports one client's exact update/gradient snapshot (.pt "bundle")
├── run_gi_nas.py             # GI-NAS attack runner
├── run_hf_gradinv.py         # HF-GradInv attack runner
├── run_adaptive_attack.py    # Adap-CTA adaptive attack runner
├── check_adaptive_protocol.py# sanity checks for the Adap-CTA protocol
├── test_Adap-CTA.py          # standalone model/dataset testing utilities
├── flcore/
│   ├── servers/              # server side: FedAvg, FedProx, SCAFFOLD, Adap-CTA (serverS_scaffold.py),
│   │                         #   privacy baselines (server_privacy.py), HE (paillier.py),
│   │                         #   reconstruction utilities (inversefed/, reconstructor.py)
│   ├── clients/              # client side: clientbase, clientscaffold, clientS_scaffold, DP/HE clients
│   ├── attacks/              # gradient-inversion attacks
│   │   ├── adaptive.py       #   Adap-CTA adaptive one-step surrogate attack (this paper)
│   │   ├── gi_nas/           #   GI-NAS (TIFS 2025) adapter + official snapshot (see its README.md)
│   │   └── hf_gradinv/       #   HF-GradInv (AAAI 2024) project adapter (see its README.md)
│   └── trainmodel/           # models: ResNet, VGG, MobileNet, LSTM, CNN, etc.
├── utils/                    # data loading, DLG evaluation, Paillier implementation, result/memory helpers
└── privacy/                  # example attack outputs: reconstructed images + reconstruction_metrics.txt
```

## Supported FL protocols (`-algo` in `main.py`)

| Algorithm     | Role                              | Where |
|---------------|-----------------------------------|-------|
| `Adap-CTA`    | **Proposed protocol** (default)   | `flcore/servers/serverS_scaffold.py`, `flcore/clients/clientS_scaffold.py` |
| `FedAvg`      | Baseline                          | `flcore/servers/serveravg.py` |
| `FedProx`     | Baseline                          | `flcore/servers/serverprox.py` |
| `SCAFFOLD`    | Baseline                          | `flcore/servers/serverscaffold.py` |
| `DPFedAvg`    | Differential-privacy defense      | `flcore/servers/serverdpavg.py`, `server_privacy.py` |
| `HEFedAvg`    | Homomorphic-encryption defense (Paillier) | `flcore/servers/serverheavg.py`, `paillier.py` |
| `FedCEO`      | Privacy baseline                  | `flcore/servers/server_privacy.py` |
| `AdapLDP-FL`  | Local-DP baseline                 | `flcore/servers/server_privacy.py` |
| `FedPVR`, `FedMut`, `exp` | Additional baselines  | `flcore/servers/` |

## Gradient-inversion attacks

All three attacks consume the same **bundle**: a `.pt` file exported by
`export_attack_bundle.py` containing the victim model state, labels, and the
target update/gradient — i.e., exactly what the server would see. Attack
quality is reported as MSE/PSNR against the ground-truth batch (kept
evaluation-only in the bundle), plus SSIM utilities in
`flcore/servers/inversefed/`.

```powershell
# 1) Export a client's update snapshot
python export_attack_bundle.py --client-data <client.npz> `
  --output privacy/gradient_bundles/client_0.pt --model resnet `
  --num-classes 10 --batch-size 4 --checkpoint <checkpoint.pt>

# 2a) Adap-CTA adaptive attack (this paper)
python run_adaptive_attack.py --bundle privacy/gradient_bundles/client_0.pt `
  --model resnet --device cuda --iterations 800 --lr 0.1

# 2b) GI-NAS (upstream-scale run; use --search-size 1 --iterations 1 as a wiring test)
python run_gi_nas.py --bundle privacy/gradient_bundles/client_0.pt `
  --device cuda --search-size 5000 --iterations 30000

# 2c) HF-GradInv (project adapter; see flcore/attacks/hf_gradinv/README.md for threat-model notes)
python run_hf_gradinv.py --bundle privacy/gradient_bundles/client_0.pt `
  --device cuda --iterations 10000 --stages 4 --restarts 3
```

## Getting started

Requirements: Python 3, `torch`, `torchvision`, `numpy`, `scipy`,
`scikit-learn`, `matplotlib`, `wandb`.

1. **Prepare datasets** — `dataset/` holds only preprocessing scripts. Generate
   the raw data first (e.g. `python dataset/generate_cifar10.py`), producing
   `dataset/Cifar10/rawdata`, `dataset/Cifar100/rawdata` (a CIFAR-20
   coarse-label variant is also used), `dataset/Shakespeare/rawdata`.
2. **Train a protocol** (from `system/system/`):

   ```powershell
   python main.py -dev cuda -did 0 -data Cifar10 -nb 10 -m resnet -algo Adap-CTA `
     -gr 800 -nc 100 -jr 0.1 -NIID 0.2
   ```

   Key flags: `-algo` protocol (see table), `-data` dataset, `-m` model,
   `-gr` rounds, `-nc` clients, `-jr` participation ratio, `-NIID` non-IID
   degree, `-nv` noise multiplier (DP).
3. **Run attacks** on exported bundles as shown above. Outputs (reconstructed
   images, `metrics.json`) are written next to the bundle.

## Notes

- This repository contains **code only**: datasets, model checkpoints, logs
  and archives are excluded via `.gitignore` and must be regenerated locally.
- `system/system/privacy/` stores example reconstruction outputs committed for
  reference (images + `reconstruction_metrics.txt`).
- `main.py` is derived from PFLlib (GPL-2.0 header retained); `flcore/attacks/gi_nas/official/`
  is a snapshot of [GI-NAS](https://github.com/cswbyu/GI-NAS) (unlicensed
  upstream — preserve attribution before redistributing). See the READMEs in
  `flcore/attacks/gi_nas/` and `flcore/attacks/hf_gradinv/` for per-attack
  implementation and threat-model caveats.
