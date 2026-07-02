"""
FCD Epilepsy MRI: Registration & Template Explorer
Dataset: OpenNeuro ds004199 (Schuch et al., 2023)

Run with:  streamlit run app.py
"""
import os

import numpy as np
import nibabel as nib
import streamlit as st

from mriapp.data import load_participants, download_subject
from mriapp.registration import register_ants, register_sitk, build_group_template
from mriapp.viz import get_slice, normalize, overlay_rgb
from mriapp.diffusion import extract_slices, train_toy_ddpm, sample_images

st.set_page_config(page_title="FCD Epilepsy MRI Explorer", layout="wide")

st.title("FCD Epilepsy MRI: Registration & Template Explorer")
st.caption(
    "Dataset: OpenNeuro ds004199 -- 85 FCD-II patients + 85 healthy controls "
    "(Schuch et al., 2023, Scientific Data)"
)

# ---------------------------------------------------------------------------
# Sidebar: subject list
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Dataset")
    with st.spinner("Loading participants.tsv..."):
        participants = load_participants()

    group_filter = st.selectbox("Group", ["fcd", "control"], index=0)
    subset = participants[participants["group"] == group_filter]
    subject_ids = subset["participant_id"].tolist()

    st.markdown(f"**{len(subject_ids)} subjects** in `{group_filter}` group")
    st.dataframe(subset.head(20), height=200)

    st.divider()
    st.caption(
        "Data pulled on demand from OpenNeuro's public S3 bucket "
        "(anonymous access, CC0 license). Downloads are cached to disk "
        "under `data/` so repeat runs are fast."
    )

tab_overview, tab_reg, tab_template, tab_diffusion = st.tabs(
    ["Subject Overview", "Intra-subject Registration", "Group Template", "Diffusion Demo (toy)"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview -- browse a single subject's raw T1 / FLAIR / ROI
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Pick a subject to inspect")
    sub_id = st.selectbox("Subject", subject_ids, key="overview_subject")

    if st.button("Download this subject's T1 / FLAIR / ROI"):
        with st.spinner(f"Downloading {sub_id}..."):
            paths = download_subject(sub_id)
        st.session_state["overview_paths"] = paths
        st.success("Downloaded.")

    if "overview_paths" in st.session_state:
        paths = st.session_state["overview_paths"]
        col1, col2 = st.columns(2)

        if paths["t1"]:
            t1_vol = nib.load(paths["t1"]).get_fdata()
            with col1:
                st.write("**T1**", t1_vol.shape)
                idx = st.slider("T1 axial slice", 0, t1_vol.shape[2] - 1, t1_vol.shape[2] // 2)
                st.image(normalize(get_slice(t1_vol, 2, idx)), width=350)

        if paths["flair"]:
            flair_vol = nib.load(paths["flair"]).get_fdata()
            roi_vol = nib.load(paths["roi"]).get_fdata() if paths["roi"] else None
            with col2:
                st.write("**FLAIR** (+ lesion ROI if published)", flair_vol.shape)
                idx2 = st.slider("FLAIR axial slice", 0, flair_vol.shape[2] - 1, flair_vol.shape[2] // 2)
                base = get_slice(flair_vol, 2, idx2)
                overlay = get_slice(roi_vol, 2, idx2) if roi_vol is not None else None
                st.image(overlay_rgb(base, overlay), width=350)
                if roi_vol is None:
                    st.caption("No lesion ROI published for this subject (op=0 / unconfirmed histopathology).")

# ---------------------------------------------------------------------------
# Tab 2: Intra-subject registration -- FLAIR -> T1, ANTs vs SimpleITK
# ---------------------------------------------------------------------------
with tab_reg:
    st.subheader("Register FLAIR to T1 (same subject)")
    st.caption(
        "The lesion ROI is published in FLAIR space -- this is the registration "
        "you'd need before overlaying it on a T1 or warping it into template/atlas space."
    )

    sub_id_reg = st.selectbox("Subject", subject_ids, key="reg_subject")
    engine = st.radio("Engine", ["ANTs", "SimpleITK", "Both (compare)"], horizontal=True)
    transform_type = st.selectbox("ANTs transform type", ["Rigid", "Affine", "SyN"], index=0)

    if st.button("Run registration"):
        with st.spinner(f"Downloading {sub_id_reg}..."):
            paths = download_subject(sub_id_reg)

        if not paths["t1"] or not paths["flair"]:
            st.error("Missing T1 or FLAIR for this subject.")
        else:
            results = []
            if engine in ("ANTs", "Both (compare)"):
                with st.spinner("Running ANTs registration..."):
                    results.append(register_ants(paths["t1"], paths["flair"], transform_type))
            if engine in ("SimpleITK", "Both (compare)"):
                with st.spinner("Running SimpleITK registration..."):
                    results.append(register_sitk(paths["t1"], paths["flair"]))

            st.session_state["reg_results"] = results

    if "reg_results" in st.session_state:
        for res in st.session_state["reg_results"]:
            st.markdown(f"### {res['engine']}")
            metrics_str = f"Time: {res['elapsed_sec']:.1f}s"
            if "metric" in res:
                metrics_str += f" | Mattes MI metric: {res['metric']:.4f}"
            st.write(metrics_str)

            n_slices = res["fixed_array"].shape[-1]
            slice_idx = st.slider(
                f"Slice ({res['engine']})", 0, n_slices - 1, n_slices // 2,
                key=f"slice_{res['engine']}"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Fixed (T1)")
                st.image(normalize(get_slice(res["fixed_array"], 2, slice_idx)), width=280)
            with col2:
                st.write("Moving (FLAIR, original)")
                st.image(normalize(get_slice(res["moving_array"], 2, slice_idx)), width=280)
            with col3:
                st.write("Warped FLAIR -> T1 space")
                st.image(normalize(get_slice(res["warped_array"], 2, slice_idx)), width=280)

# ---------------------------------------------------------------------------
# Tab 3: Group template construction (ANTs only)
# ---------------------------------------------------------------------------
with tab_template:
    st.subheader("Build an unbiased group template (ANTs)")
    st.caption(
        "Uses ants.build_template() -- iterative averaging + SyN deformable "
        "registration across subjects. SimpleITK has no equivalent, so this "
        "tab always uses ANTs. This is slow: expect several minutes even for "
        "a handful of subjects on CPU-only Codespaces."
    )

    n_subjects = st.slider("Number of subjects to include", 2, 8, 4)
    iterations = st.slider("Template-building iterations", 1, 4, 2)
    selected_subs = st.multiselect(
        "Subjects", subject_ids, default=subject_ids[:n_subjects]
    )

    if st.button("Build template", disabled=len(selected_subs) < 2):
        t1_paths = []
        progress = st.progress(0.0, text="Downloading subjects...")
        for i, sub in enumerate(selected_subs):
            paths = download_subject(sub)
            if paths["t1"]:
                t1_paths.append(paths["t1"])
            progress.progress((i + 1) / len(selected_subs), text=f"Downloaded {sub}")

        with st.spinner(
            f"Building template from {len(t1_paths)} subjects "
            f"({iterations} iterations) -- this will take a while..."
        ):
            template = build_group_template(t1_paths, iterations=iterations)

        st.session_state["template_array"] = template.numpy()
        st.success("Template built.")

    if "template_array" in st.session_state:
        tmpl = st.session_state["template_array"]
        st.write("**Resulting template**", tmpl.shape)
        idx = st.slider("Template axial slice", 0, tmpl.shape[2] - 1, tmpl.shape[2] // 2)
        st.image(normalize(get_slice(tmpl, 2, idx)), width=450)

        out_path = "data/group_template.nii.gz"
        os.makedirs("data", exist_ok=True)
        nib.save(nib.Nifti1Image(tmpl, affine=np.eye(4)), out_path)
        with open(out_path, "rb") as f:
            st.download_button("Download template (.nii.gz)", f, file_name="group_template.nii.gz")

# ---------------------------------------------------------------------------
# Tab 4: Toy diffusion demo -- trained fast, from scratch, on your slices
# ---------------------------------------------------------------------------
with tab_diffusion:
    st.subheader("Toy 2D Diffusion Demo")
    st.warning(
        "This is **not** the pretrained MONAI/UK-Biobank latent diffusion "
        "model -- that one is anatomically meaningful but takes far too long "
        "to sample on CPU-only machines. This is a small pixel-space DDPM "
        "trained from scratch, in about a minute, on axial slices from "
        "subjects you've downloaded here. It demonstrates the diffusion "
        "*mechanism* (denoising from noise step by step), not a "
        "scientifically valid brain generator -- outputs will look like "
        "blurry, brain-ish blobs, not real anatomy."
    )

    diff_subs = st.multiselect(
        "Subjects to train on (more subjects = more varied slices, still fast)",
        subject_ids, default=subject_ids[:6], key="diff_subs"
    )
    img_size = st.select_slider("Image size", options=[24, 32, 48], value=32)
    train_steps = st.slider("Training steps", 50, 500, 200, step=50)

    if st.button("Download slices & train", disabled=len(diff_subs) < 3):
        t1_paths = []
        dl_progress = st.progress(0.0, text="Downloading subjects...")
        for i, sub in enumerate(diff_subs):
            paths = download_subject(sub)
            if paths["t1"]:
                t1_paths.append(paths["t1"])
            dl_progress.progress((i + 1) / len(diff_subs), text=f"Downloaded {sub}")

        with st.spinner("Extracting slices..."):
            dataset = extract_slices(t1_paths, img_size=img_size)

        if dataset is None or dataset.shape[0] < 10:
            st.error("Not enough usable slices extracted -- try selecting more subjects.")
        else:
            st.write(f"Extracted {dataset.shape[0]} slices for training.")
            train_progress = st.progress(0.0, text="Training...")
            loss_display = st.empty()

            def _progress_cb(step, total, loss):
                train_progress.progress(step / total, text=f"Step {step}/{total}")
                loss_display.text(f"loss: {loss:.4f}")

            model = train_toy_ddpm(
                dataset, img_size=img_size, num_train_steps=train_steps,
                progress_callback=_progress_cb
            )
            st.session_state["diffusion_model"] = model
            st.session_state["diffusion_img_size"] = img_size
            st.success("Training complete.")

    if "diffusion_model" in st.session_state:
        st.divider()
        n_samples = st.slider("Number of samples to generate", 1, 8, 4)
        inference_steps = st.slider("Sampling steps (fewer = faster, blurrier)", 10, 100, 30)

        if st.button("Generate samples"):
            with st.spinner("Sampling..."):
                imgs = sample_images(
                    st.session_state["diffusion_model"],
                    num_samples=n_samples,
                    img_size=st.session_state["diffusion_img_size"],
                    num_inference_steps=inference_steps,
                )
            cols = st.columns(n_samples)
            for i, col in enumerate(cols):
                with col:
                    st.image(imgs[i], width=150, caption=f"sample {i + 1}")