
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ve_core import DEFAULT_PARAMS, run_forward, choose_well_ij, prepare_phi_k
from eclipse_io import load_eclipse_phi_k_from_uploads
from schedule_tools import build_cycle_schedule, schedule_to_csv_bytes

st.set_page_config(page_title="VE+Darcy+Land Forward Predictor", layout="wide")
st.title("VE + Darcy + Land: Universal Forward Plume Predictor")

st.markdown(
    """
This app runs your **VE+Darcy+Land paper model** forward **without sg_obs**.

### Input options
**A) NPZ mode**  
Upload an NPZ containing:
- `phi` : 2D array (nx, ny)
- `k`   : 2D array (nx, ny)

**B) Eclipse mode (recommended: ZIP)**  
Upload a **ZIP** that contains your Eclipse deck files (`.DATA`, `.GRDECL`, `.INC`, etc.).  
The app will extract `PORO` and `PERMX` (or other K keyword you select) and create a 2D layer/mean map.

### Schedule options
- Upload a schedule CSV (`t`, `q`) **OR**
- Build a cyclic schedule in **ton/day** and download as CSV.

> Tip: The forward model typically behaves best when `max(|q|) ≈ 1` in model units.  
  If you use ton/day, the schedule builder normalizes by `q_ref` automatically.
"""
)

# -------------------------
# Helpers
# -------------------------

def load_npz(uploaded):
    data = np.load(uploaded)
    if "phi" not in data or "k" not in data:
        raise ValueError("NPZ must contain keys: phi and k")
    return data["phi"].astype(np.float32), data["k"].astype(np.float32)

def load_schedule_csv(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "t" not in df.columns or "q" not in df.columns:
        raise ValueError("CSV must have columns: t, q (optionally day, q_ton_day)")
    t = df["t"].to_numpy(dtype=np.float32)
    q = df["q"].to_numpy(dtype=np.float32)
    return t, q

def box_smooth_2d(x, k: int = 3):
    """
    NaN-aware box smoothing using an integral image.
    Always returns the SAME shape as input (H, W).
    Works for any H,W and any k>=1.
    """
    if x is None:
        return None

    x = np.asarray(x, dtype=np.float32)
    if k <= 1:
        return x

    # Force odd window so output shape is guaranteed (H,W)
    k = int(k)
    if (k % 2) == 0:
        k += 1
    pad = k // 2

    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0).astype(np.float32)
    w0 = valid.astype(np.float32)

    # Pad with zeros; weights handle edges correctly
    x0p = np.pad(x0, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    w0p = np.pad(w0, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)

    # Integral images with a 1-cell zero border
    S = np.pad(x0p, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    W = np.pad(w0p, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    S = np.cumsum(np.cumsum(S, axis=0), axis=1)
    W = np.cumsum(np.cumsum(W, axis=0), axis=1)

    # Window sum via 4 corners -> output is exactly (H, W)
    total = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]
    wsum  = W[k:, k:] - W[:-k, k:] - W[k:, :-k] + W[:-k, :-k]

    out = np.where(wsum > 0, total / wsum, np.nan).astype(np.float32)
    return out


def fig_imshow(arr, title, vmin=None, vmax=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    if vmin is None: vmin = float(np.nanmin(arr))
    if vmax is None: vmax = float(np.nanmax(arr))
    im = plt.imshow(arr, origin="lower", vmin=vmin, vmax=vmax)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def fig_schedule(t, q, tidx=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title("Schedule q(t)")
    plt.plot(t, q)
    if tidx is not None and 0 <= tidx < len(t):
        plt.axvline(t[tidx], linestyle="--")
    plt.xlabel("t")
    plt.ylabel("q")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def fig_timeseries(t, y, ylab, title):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    plt.plot(t, y)
    plt.xlabel("t")
    plt.ylabel(ylab)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("1) Input mode")
    mode = st.radio("Choose input mode", ["NPZ (phi/k)", "Eclipse (ZIP / deck files)"], index=1)

    st.divider()
    st.header("2) Upload inputs")
    if mode.startswith("NPZ"):
        up_npz = st.file_uploader("phi/k NPZ", type=["npz"])
        eclipse_uploads = []
    else:
        eclipse_uploads = st.file_uploader(
            "Eclipse deck upload (ZIP recommended, or multiple files)",
            type=["zip", "data", "grdecl", "inc", "txt"],
            accept_multiple_files=True,
        )
        up_npz = None

    st.divider()
    st.header("3) Eclipse extraction options")
    # Only used in Eclipse mode
    extract_mode = st.selectbox("2D extraction", ["layer", "mean"], index=0)
    layer = st.number_input("Layer index (if layer mode)", value=0, step=1)
    kkey = st.text_input("K keyword (e.g., PERMX)", value="PERMX")

    st.divider()
    st.header("4) Well location")
    well_mode = st.selectbox("Well placement", ["max_k", "center", "manual"], index=0)
    manual_i = st.number_input("manual i", value=0, step=1)
    manual_j = st.number_input("manual j", value=0, step=1)

    st.divider()
    st.header("5) Schedule")
    sched_mode = st.radio("Schedule source", ["Upload CSV (t,q)", "Build cyclic schedule (ton/day)"], index=1)

    up_csv = None
    if sched_mode.startswith("Upload"):
        up_csv = st.file_uploader("schedule CSV", type=["csv"])
        q_scale = st.number_input("Schedule scale factor (multiplies q)", value=1.0)
        normalize_q = st.checkbox("Normalize q to max(|q|)=1", value=False)
    else:
        total_days = st.number_input("Total days", value=365, step=10)
        dt_days = st.number_input("dt (days per step)", value=1, step=1)
        inj_days = st.number_input("Injection days per cycle", value=30, step=1)
        shut1 = st.number_input("Shut-in days after injection", value=5, step=1)
        prod_days = st.number_input("Production days per cycle", value=30, step=1)
        shut2 = st.number_input("Shut-in days after production", value=5, step=1)
        inj_rate = st.number_input("Injection rate (ton/day)", value=1000.0, step=100.0)
        prod_rate = st.number_input("Production rate (ton/day)", value=800.0, step=100.0)
        q_ref = st.number_input("Reference rate q_ref (ton/day, for normalization)", value=1000.0, step=100.0)

        s = build_cycle_schedule(
            total_days=int(total_days),
            dt_days=int(dt_days),
            inj_days=int(inj_days),
            shut1_days=int(shut1),
            prod_days=int(prod_days),
            shut2_days=int(shut2),
            inj_rate_ton_day=float(inj_rate),
            prod_rate_ton_day=float(prod_rate),
            q_ref_ton_day=float(q_ref),
        )
        st.caption(f"Built schedule: Nt={len(s.t)} steps | max(|q_model|)={float(np.max(np.abs(s.q_model))):.3f}")

        st.download_button(
            "Download schedule CSV (t,q + ton/day columns)",
            data=schedule_to_csv_bytes(s, use_model_q=True),
            file_name="schedule_cycles_ton_day.csv",
        )

    st.divider()
    st.header("6) Paper parameters")
    use_defaults = st.checkbox("Use paper-calibrated defaults", value=True)
    params = dict(DEFAULT_PARAMS)
    if not use_defaults:
        # expose the important ones (you can add more)
        params["D0"] = st.number_input("D0", value=float(params["D0"]))
        params["alpha_p"] = st.number_input("alpha_p", value=float(params["alpha_p"]))
        params["src_amp"] = st.number_input("src_amp", value=float(params["src_amp"]))
        params["prod_frac"] = st.number_input("prod_frac", value=float(params["prod_frac"]))
        params["Swr"] = st.number_input("Swr", value=float(params["Swr"]))
        params["Sgr_max"] = st.number_input("Sgr_max", value=float(params["Sgr_max"]))
        params["C_L"] = st.number_input("C_L", value=float(params["C_L"]))
        params["hc"] = st.number_input("hc", value=float(params["hc"]))
        params["mob_exp"] = st.number_input("mob_exp", value=float(params["mob_exp"]))
        params["anisD"] = st.number_input("anisD", value=float(params["anisD"]))

    st.divider()
    st.header("7) Visualization")
    thr_area = st.slider("Area threshold (Sg > thr)", 0.0, 0.5, 0.05, 0.01)
    smooth_display = st.checkbox("Smooth display (does NOT change saved output)", value=True)
    smooth_k = int(st.slider("Smooth kernel size", 1, 31, 5))
if smooth_k % 2 == 0:
    smooth_k += 1  

    st.divider()
    run_btn = st.button("Run forward prediction", type="primary")


# -------------------------
# Load inputs
# -------------------------
phi = k = None
meta_info = {}

try:
    if mode.startswith("NPZ"):
        if up_npz is None:
            st.info("Upload a phi/k NPZ to begin.")
            st.stop()
        phi, k = load_npz(up_npz)
        meta_info["source"] = up_npz.name
    else:
        if not eclipse_uploads:
            st.info("Upload an Eclipse ZIP or deck files to begin.")
            st.stop()
        r = load_eclipse_phi_k_from_uploads(
            eclipse_uploads,
            mode=extract_mode,
            layer=int(layer),
            kkey=str(kkey).strip() if str(kkey).strip() else "PERMX",
        )
        phi, k, meta_info = r.phi2d, r.k2d, r.meta

except Exception as e:
    st.error(f"Failed to read inputs: {e}")
    if meta_info.get("warnings"):
        with st.expander("Reader warnings"):
            st.write(meta_info["warnings"])
    st.stop()

# Schedule
try:
    if sched_mode.startswith("Upload"):
        if up_csv is None:
            st.info("Upload a schedule CSV (t,q) OR switch to schedule builder.")
            st.stop()
        t, q = load_schedule_csv(up_csv)
        q = q.astype(np.float32) * np.float32(q_scale)
        if normalize_q:
            m = float(np.max(np.abs(q))) if q.size else 1.0
            if m > 0:
                q = (q / m).astype(np.float32)
    else:
        t = s.t.astype(np.float32)
        q = s.q_model.astype(np.float32)

except Exception as e:
    st.error(f"Failed to read schedule: {e}")
    st.stop()

# -------------------------
# Preview inputs
# -------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("Input: porosity (phi)")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}"))
with colB:
    st.subheader("Input: permeability (k)")
    st.pyplot(fig_imshow(k, f"k | shape={k.shape}"))

if meta_info.get("warnings"):
    with st.expander("Eclipse reader warnings"):
        for w in meta_info["warnings"]:
            st.write("- " + str(w))

# well selection preview
try:
    _, k_norm, mask = prepare_phi_k(phi, k)
    if well_mode == "manual":
        well_ij = (int(manual_i), int(manual_j))
    else:
        well_ij = None
    wi, wj = choose_well_ij(k_norm, mask, well_mode, ij=well_ij)
    st.caption(f"Selected well (i,j)=({wi},{wj}) | active cells={int(mask.sum())}")
except Exception as e:
    st.error(f"Input fields invalid: {e}")
    st.stop()

st.subheader("Schedule")
st.pyplot(fig_schedule(t, q, tidx=0))

if not run_btn:
    st.stop()

# -------------------------
# Run forward model
# -------------------------
with st.spinner("Running VE+Darcy+Land forward model..."):
    try:
        res = run_forward(
            phi=phi,
            k=k,
            t=t,
            q=q,
            params=params,
            well_mode=well_mode,
            well_ij=(int(manual_i), int(manual_j)) if well_mode == "manual" else None,
            return_pressure=True,
            thr_area=float(thr_area),
        )
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

st.success("Done.")

Nt = len(res.sg_list)
tidx = st.slider("Select timestep (tidx)", 0, max(0, Nt - 1), min(0, Nt - 1))

sg = res.sg_list[tidx]
p = res.p_list[tidx] if res.p_list is not None else None

if smooth_display:
    sg_show = box_smooth_2d(sg, k=smooth_k)
    p_show = box_smooth_2d(p, k=smooth_k) if p is not None else None
else:
    sg_show = sg
    p_show = p

left, right = st.columns(2)
with left:
    st.subheader(f"Sg predicted | tidx={tidx}")
    st.pyplot(fig_imshow(sg_show, f"Sg predicted | tidx={tidx}", vmin=0.0, vmax=1.0))
with right:
    st.subheader(f"Pressure surrogate | tidx={tidx}")
    if p_show is None:
        st.write("Pressure output disabled.")
    else:
        st.pyplot(fig_imshow(p_show, f"p predicted | tidx={tidx}"))

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("q(t)")
    st.pyplot(fig_schedule(res.t, res.q, tidx=tidx))
with col2:
    st.subheader("Plume area")
    st.pyplot(fig_timeseries(res.t, res.area, "area (cells)", "Area time series"))
with col3:
    st.subheader("Equivalent radius")
    st.pyplot(fig_timeseries(res.t, res.r_eq, "r_eq (cells)", "Equivalent radius time series"))

st.subheader("Download results")
out_npz = io.BytesIO()
sg_stack = np.stack([np.nan_to_num(s, nan=0.0) for s in res.sg_list], axis=0).astype(np.float32)
np.savez_compressed(out_npz, sg=sg_stack, t=res.t, q=res.q, area=res.area, r_eq=res.r_eq)
st.download_button("Download predicted Sg (NPZ)", data=out_npz.getvalue(), file_name="sg_predicted.npz")

out_csv = pd.DataFrame({"t": res.t, "q": res.q, "area": res.area, "r_eq": res.r_eq}).to_csv(index=False).encode("utf-8")
st.download_button("Download time series (CSV)", data=out_csv, file_name="plume_timeseries.csv")
