# HF-GradInv integration

This directory contains a project-facing implementation of *High-Fidelity
Gradient Inversion in Distributed Learning* (Ye et al., AAAI 2024,
DOI: `10.1609/aaai.v38i18.29975`). The original paper uses evolutionary
label inference followed by stepwise gradient matching. The adapter keeps the
same two-stage idea while consuming the model/gradient bundles already used by
the project.

For portability, the label stage uses the paper's last-classifier-gradient
initialization together with an auxiliary feature/probability estimate (and
records the feature CV); it is not a dependency on the upstream notebook
training harness. Results should therefore be described as the project
adapter's HF-GradInv implementation and validated against the official code
on at least one reference setting when publishing quantitative claims.

The implementation is intentionally independent of the upstream
`breaching` harness. It supports image classifiers with a final `nn.Linear`
layer and currently expects three-channel images. An optional disjoint public
auxiliary image set (`[N,C,H,W]`, values in `[0,1]`) enables the
CV-assisted label initialization. The default runner does not pass the bundle
labels to the attack; those labels are used only for post-attack metrics.

## Workflow

Export a snapshot with the shared standalone exporter:

```powershell
python export_attack_bundle.py --client-data <client.npz> `
  --output privacy/gradient_bundles/client_0.pt --model resnet `
  --num-classes 10 --batch-size 4 --checkpoint <checkpoint.pt>
```

Then run HF-GradInv with a short wiring-test budget:

```powershell
python run_hf_gradinv.py `
  --bundle privacy/gradient_bundles/client_0.pt `
  --device cuda --iterations 4 --stages 2
```

For a paper experiment, increase the optimization budget (for example,
`--iterations 10000` or more), use multiple restarts, and provide a public,
disjoint auxiliary set:

```powershell
python run_hf_gradinv.py `
  --bundle privacy/gradient_bundles/client_0.pt `
  --auxiliary-data path/to/public_cifar10_images.pt `
  --device cuda --iterations 10000 --stages 4 --restarts 3
```

For a controlled cross-method table, use `--use-ground-truth-labels` so
HF-GradInv and GI-NAS receive the same known label multiset. Report
the default inferred-label mode separately as the HF-GradInv label-inference
ablation; mixing these two threat models in one ranking is not a fair
comparison.

The attack consumes a single-batch gradient snapshot. The normal FL code
uploads multi-step client models, so applying HF-GradInv directly to a model
difference requires a separately stated update-to-gradient threat model.
