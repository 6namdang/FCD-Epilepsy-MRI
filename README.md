# FCD Epilepsy MRI: Registration & Template Explorer

A Streamlit app for exploring MRI registration and group template
construction on the OpenNeuro **ds004199** dataset (85 people with
epilepsy due to focal cortical dysplasia type II + 85 healthy controls;
Schuch et al., 2023, *Scientific Data*).

## What it does

- **Subject Overview** -- browse a subject's raw T1 / FLAIR, with the
  lesion ROI overlaid on FLAIR when published.
- **Intra-subject Registration** -- register FLAIR to T1 for one subject,
  using ANTs, SimpleITK, or both side by side (timing + similarity metric).
- **Group Template** -- build an unbiased anatomical template from
  multiple T1 volumes using `ants.build_template()` (iterative SyN
  registration + averaging).
- **Diffusion Demo (toy)** -- trains a small pixel-space DDPM from scratch,
  in about a minute on CPU, on axial slices from your downloaded subjects.
  This is a mechanism demo, not a scientifically valid generator -- see the
  in-app warning for why it's not a true latent diffusion model.

Data is downloaded on demand from OpenNeuro's public S3 bucket
(anonymous access, CC0 license) and cached under `data/`.

## Setup in GitHub Codespaces

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Codespaces will detect the forwarded port and offer to open it in the
browser (or use the "Ports" tab to open it manually).

## Notes / known constraints

- **CPU only, no GPU acceleration.** ANTs SyN registration and template
  building are computationally heavy. Rigid/Affine registration on one
  subject takes seconds to ~1 minute; template building across 4+
  subjects with SyN can take several minutes to tens of minutes
  depending on the Codespace's machine size.
- **`antspyx` install** can occasionally hit numpy version conflicts
  depending on the Codespaces base image. If `pip install -r
  requirements.txt` fails on antspyx, try pinning numpy first
  (`pip install "numpy<2"`) and reinstalling.
- **Template building is ANTs-only.** SimpleITK has no built-in
  equivalent to `ants.build_template()`.
- **The diffusion tab is a toy, not a real LDM.** A real latent diffusion
  model (e.g. the MONAI/Pinaya et al. UK Biobank model) trains a VAE to
  compress into latent space first, and generates full 3D volumes -- too
  slow to sample on CPU-only machines. This tab trains a small pixel-space
  DDPM directly on 2D slices instead, purely to demonstrate the mechanism.
- **Masks are in FLAIR space**, not T1 space -- that's why the
  registration tab exists as a separate step before any
  template/atlas alignment work involving lesion masks.
- Not all subjects have a published lesion ROI (only those with
  confirmed histopathology / completed surgery). The Overview tab
  will say so when it's missing.
- **Package name note:** the local helper package is named `mriapp`
  (not `utils`) deliberately -- `utils` collides with an unrelated
  PyPI package of the same name and will silently shadow your local code.

## Project structure

```
app.py                  # Streamlit entry point (4 tabs)
mriapp/
  __init__.py
  data.py                # S3 download + participants.tsv loading
  registration.py         # ANTs + SimpleITK registration, template building
  viz.py                  # slice extraction, normalization, mask overlay
  diffusion.py             # toy 2D DDPM: slice extraction, train, sample
requirements.txt
```