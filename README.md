
# VE + Darcy + Land (Universal Forward Predictor)

Streamlit app to run the **VE+Darcy+Land** forward model (no `sg_obs` required).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
- Push these files to GitHub.
- In Streamlit Cloud, set:
  - **Main file path**: `app.py`
  - **Python version**: 3.10+ recommended

## Inputs

### Option A: NPZ
NPZ keys:
- `phi` (2D)
- `k` (2D)

### Option B: Eclipse (ZIP recommended)
Upload a ZIP that contains your Eclipse deck files (`.DATA`, `.GRDECL`, `.INC`, etc.).  
The reader extracts `PORO` and a permeability keyword (default `PERMX`), then creates a 2D map (`layer` or `mean`).

## Schedule

### Upload CSV
CSV must contain columns:
- `t`
- `q` (model units; typically keep max(|q|) around 1)

### Build cyclic schedule (ton/day)
Use the UI to define:
- injection days / shut-in / production days / shut-in
- rates in ton/day
- reference rate `q_ref` for normalization

The app can download a schedule CSV containing both physical and model units.

## Notes
- If you see warnings about missing INCLUDE files, upload a ZIP containing the full deck directory.
- If a deck contains `T/F` tokens inside numeric arrays, the reader maps them to `1/0`.
