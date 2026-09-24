# GI-NAS integration

This directory adapts the official two-stage GI-NAS gradient inversion attack
to this federated-learning project. The original source snapshot is stored in
`official/`; `attack.py` is a small project-facing adapter and `bundle.py`
defines a reproducible model/gradient snapshot format.

Upstream source: <https://github.com/cswbyu/GI-NAS>, commit `6193dd1`.

Paper: *GI-NAS: Boosting Gradient Inversion Attacks Through Adaptive Neural
Architecture Search*, IEEE Transactions on Information Forensics and Security,
2025, DOI: `10.1109/TIFS.2025.3589127`.

The upstream repository did not contain a license file when this source
snapshot was taken. Preserve this attribution and check redistribution terms
before publishing a derived release.

## Workflow

First export one exact client batch gradient with the shared exporter:

```powershell
python export_attack_bundle.py --client-data <client.npz> `
  --output privacy/gradient_bundles/client_0.pt --model resnet `
  --num-classes 10 --batch-size 4 --checkpoint <checkpoint.pt>
```

Add `--convert-relu` when reproducing the upstream CIFAR-10 script's
optional GION-compatible ReLU-to-Sigmoid victim conversion. It is off by
default so the attack measures the actual FL backbone.

The exporter writes a `.pt` bundle. Ground-truth images are included only for
evaluation and figures; the reconstruction routine consumes the victim model,
labels, and target gradient.

The same bundle can also be used by `run_hf_gradinv.py`.

Then run the official-size experiment:

```powershell
python run_gi_nas.py `
  --bundle privacy/gradient_bundles/client_0.pt `
  --device cuda --search-size 5000 --iterations 30000
```

For a quick wiring test, use `--search-size 1 --iterations 1`; this is not a
paper-quality result.
