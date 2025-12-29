
"""
eclipse_io.py

Robust, Streamlit-friendly reader to extract 2D phi/k maps from Eclipse-style inputs.

Supports:
- Upload a ZIP that contains .DATA and its INCLUDE files, or a .GRDECL.
- Upload individual files (DATA/GRDECL/INC) via Streamlit.

Output:
- phi2d (nx, ny) float32 (masked inactive -> NaN)
- k2d   (nx, ny) float32 (masked inactive -> NaN)
- meta dict with nx, ny, nz, mode, layer, source, warnings

Notes:
- Handles Eclipse repeat notation: '10*0.25'
- Handles logical tokens in numeric blocks: T/F -> 1/0
- Handles "n*" (missing value) as NaN
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import io
import re
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# Token parsing utilities
# -------------------------

_RE_REPEAT = re.compile(r"^(\d+)\*(.*)$")

def _strip_comments(line: str) -> str:
    # Eclipse comments are commonly: '--' or '#'
    line = line.split("--", 1)[0]
    line = line.split("#", 1)[0]
    return line.strip()

def _tok_to_float(tok: str) -> float:
    t = tok.strip().upper()
    if t in ("T", "TRUE"):
        return 1.0
    if t in ("F", "FALSE"):
        return 0.0
    # Accept D exponent (Fortran)
    t = t.replace("D", "E")
    return float(t)

def _parse_repeat_token(tok: str) -> List[float]:
    """
    Supports:
      - '10*0.25'  -> [0.25]*10
      - '10*F'     -> [0.0]*10
      - '10*T'     -> [1.0]*10
      - '10*'      -> [nan]*10  (value omitted)
      - '1e-3'     -> [0.001]
    """
    tok = tok.strip()
    if not tok:
        return []
    m = _RE_REPEAT.match(tok)
    if m:
        n = int(m.group(1))
        vraw = m.group(2).strip()
        if vraw == "":
            return [float("nan")] * n
        v = _tok_to_float(vraw)
        return [v] * n
    return [_tok_to_float(tok)]

def _parse_numeric_block(lines: List[str], start_idx: int) -> Tuple[np.ndarray, int]:
    """
    Parse numbers from lines starting at start_idx until encountering a '/'.
    Returns (values, end_idx_inclusive).
    """
    vals: List[float] = []
    i = start_idx
    while i < len(lines):
        s = _strip_comments(lines[i])
        if not s:
            i += 1
            continue

        if "/" in s:
            head = s.split("/", 1)[0].strip()
            if head:
                for tok in head.replace(",", " ").split():
                    vals.extend(_parse_repeat_token(tok))
            return np.array(vals, dtype=np.float32), i

        for tok in s.replace(",", " ").split():
            vals.extend(_parse_repeat_token(tok))
        i += 1

    raise RuntimeError("Numeric block did not terminate with '/'")

def _find_keyword(lines: List[str], keyword: str) -> Optional[int]:
    kw = keyword.upper().strip()
    for i, raw in enumerate(lines):
        if _strip_comments(raw).upper() == kw:
            return i
    return None

def _read_keyword_array(lines: List[str], keyword: str) -> Optional[np.ndarray]:
    idx = _find_keyword(lines, keyword)
    if idx is None:
        return None
    arr, _ = _parse_numeric_block(lines, idx + 1)
    return arr

def _read_specgrid(lines: List[str]) -> Tuple[int, int, int]:
    idx = _find_keyword(lines, "SPECGRID")
    if idx is None:
        idx = _find_keyword(lines, "DIMENS")
    if idx is None:
        raise RuntimeError("Could not find SPECGRID or DIMENS in deck/GRDECL.")
    block, _ = _parse_numeric_block(lines, idx + 1)
    if block.size < 3:
        raise RuntimeError("SPECGRID/DIMENS block has < 3 numbers.")
    nx, ny, nz = int(block[0]), int(block[1]), int(block[2])
    return nx, ny, nz

def _to_2d(arr3d: np.ndarray, nx: int, ny: int, nz: int, mode: str, layer: int) -> np.ndarray:
    a = arr3d.reshape((nz, ny, nx))  # Eclipse ordering often (K, J, I)
    a = np.transpose(a, (2, 1, 0))   # -> (I, J, K)
    if mode == "layer":
        k = int(np.clip(layer, 0, nz - 1))
        return a[:, :, k]
    if mode == "mean":
        return np.nanmean(a, axis=2)
    raise ValueError("mode must be 'layer' or 'mean'")


# -------------------------
# INCLUDE flattening
# -------------------------

def _extract_include_path(line: str) -> Optional[str]:
    """
    Accept:
      INCLUDE
      'file.INC' /
    or:
      INCLUDE 'file.INC' /
    or (rare):
      INCLUDE file.INC /
    """
    s = _strip_comments(line)
    m = re.search(r"'([^']+)'", s)
    if m:
        return m.group(1)
    toks = s.split()
    # e.g. INCLUDE file.inc /
    if len(toks) >= 2 and toks[0].upper() == "INCLUDE":
        cand = toks[1]
        cand = cand.replace('"', "").strip()
        # strip trailing slash
        cand = cand.replace("/", "").strip()
        return cand if cand else None
    return None

def flatten_deck_with_includes(entry_path: Path, max_depth: int = 30) -> Tuple[List[str], List[str]]:
    """
    Return (flattened_lines, warnings).
    If an INCLUDE has no path or missing file, it is skipped with warning (instead of failing).
    """
    visited = set()
    warnings: List[str] = []

    def _flatten(p: Path, depth: int) -> List[str]:
        if depth > max_depth:
            raise RuntimeError("Too deep INCLUDE nesting (possible loop).")
        p = p.resolve()
        key = str(p)
        if key in visited:
            return []
        visited.add(key)

        try:
            txt = p.read_text(errors="ignore")
        except Exception as e:
            warnings.append(f"Could not read file: {p.name} ({e})")
            return []
        lines = txt.splitlines()
        out: List[str] = []
        i = 0
        while i < len(lines):
            s_up = _strip_comments(lines[i]).upper()
            # INCLUDE can appear as keyword alone, or with a path on same line
            if s_up.startswith("INCLUDE"):
                # try same line first
                inc_rel = _extract_include_path(lines[i])
                j = i + 1
                # if not on same line, search forward for the first line containing a path
                if inc_rel is None:
                    while j < len(lines) and not _strip_comments(lines[j]):
                        j += 1
                    if j < len(lines):
                        inc_rel = _extract_include_path(lines[j])
                if inc_rel is None:
                    warnings.append(f"INCLUDE without path near line {i+1} in {p.name} (skipped)")
                    # skip until the line containing '/'
                    k = i
                    while k < len(lines) and "/" not in lines[k]:
                        k += 1
                    i = k + 1
                    continue

                inc_path = (p.parent / inc_rel).resolve()
                if not inc_path.exists():
                    warnings.append(f"Missing INCLUDE file '{inc_rel}' referenced in {p.name} (skipped)")
                else:
                    out.extend(_flatten(inc_path, depth + 1))

                # advance i to after the INCLUDE statement terminator '/'
                k = i
                # include statement usually ends on the same line or the next with '/'
                while k < len(lines) and "/" not in lines[k]:
                    k += 1
                i = k + 1
                continue

            out.append(lines[i])
            i += 1
        return out

    flat = _flatten(entry_path, 0)
    return flat, warnings


# -------------------------
# Streamlit helpers
# -------------------------

@dataclass
class EclipseLoadResult:
    phi2d: np.ndarray
    k2d: np.ndarray
    meta: Dict[str, object]


def _save_upload_to_dir(upload, out_dir: Path) -> Path:
    data = upload.getvalue()
    p = out_dir / upload.name
    p.write_bytes(data)
    return p


def load_eclipse_phi_k_from_uploads(
    uploads: List[object],
    mode: str = "layer",
    layer: int = 0,
    kkey: str = "PERMX",
) -> EclipseLoadResult:
    """
    uploads: list of Streamlit UploadedFile objects. Can include:
      - a single .zip containing deck files
      - .DATA and/or .GRDECL plus .INC
    """
    tmp = Path(".eclipse_tmp")
    if tmp.exists():
        # keep it small: wipe each run
        for child in tmp.iterdir():
            try:
                if child.is_file():
                    child.unlink()
                else:
                    import shutil
                    shutil.rmtree(child)
            except Exception:
                pass
    tmp.mkdir(exist_ok=True)

    paths: List[Path] = []
    warnings: List[str] = []

    # If a zip is uploaded, extract it
    zips = [u for u in uploads if u.name.lower().endswith(".zip")]
    if zips:
        if len(zips) > 1:
            warnings.append("Multiple ZIPs uploaded; using the first one.")
        z = zips[0]
        zpath = _save_upload_to_dir(z, tmp)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmp)
        # gather all extracted files
        for p in tmp.rglob("*"):
            if p.is_file():
                paths.append(p)
    else:
        for u in uploads:
            paths.append(_save_upload_to_dir(u, tmp))

    # Prefer GRDECL if present (often contains PORO/PERMX directly)
    grdecls = [p for p in paths if p.suffix.lower() == ".grdecl"]
    datas = [p for p in paths if p.suffix.lower() == ".data"]

    source = None
    lines: List[str] = []

    if grdecls:
        source = grdecls[0]
        try:
            lines = source.read_text(errors="ignore").splitlines()
        except Exception as e:
            raise RuntimeError(f"Could not read GRDECL: {source.name} ({e})")
    elif datas:
        source = datas[0]
        flat, warn = flatten_deck_with_includes(source)
        warnings.extend(warn)
        lines = flat
    else:
        raise RuntimeError("No .GRDECL or .DATA found. Upload a ZIP (recommended) or the deck files.")

    # Parse grid
    nx, ny, nz = _read_specgrid(lines)
    n = nx * ny * nz

    phi = _read_keyword_array(lines, "PORO")
    kx = _read_keyword_array(lines, kkey)
    act = _read_keyword_array(lines, "ACTNUM")

    if phi is None:
        raise RuntimeError("PORO keyword not found (in GRDECL or flattened deck).")
    if kx is None:
        raise RuntimeError(f"{kkey} keyword not found (try PERMX or PERMY/PERMZ).")

    if phi.size != n:
        raise RuntimeError(f"PORO has {phi.size} values, expected {n} (nx*ny*nz).")
    if kx.size != n:
        raise RuntimeError(f"{kkey} has {kx.size} values, expected {n} (nx*ny*nz).")

    # Apply ACTNUM if present
    if act is not None and act.size == n:
        mask = (act.reshape((n,)) > 0)
        phi = np.where(mask, phi, np.nan).astype(np.float32)
        kx  = np.where(mask, kx,  np.nan).astype(np.float32)

    phi2 = _to_2d(phi, nx, ny, nz, mode, layer).astype(np.float32)
    k2   = _to_2d(kx,  nx, ny, nz, mode, layer).astype(np.float32)

    meta: Dict[str, object] = {
        "nx": nx, "ny": ny, "nz": nz,
        "mode": mode, "layer": int(layer), "kkey": kkey,
        "source": str(source.name) if source else "",
        "warnings": warnings,
    }
    return EclipseLoadResult(phi2d=phi2, k2d=k2, meta=meta)
