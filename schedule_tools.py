
"""
schedule_tools.py

Utilities to build user-friendly schedules in physical units (ton/day),
then convert to the model's dimensionless q(t) by dividing by a reference rate.

The VE forward model only needs:
- t: array (timesteps) or time
- q: signed rate per step (positive injection, negative production)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Schedule:
    t: np.ndarray          # integer timestep index
    day: np.ndarray        # physical time in days
    q_ton_day: np.ndarray  # physical signed rate
    q_model: np.ndarray    # scaled rate used by the model


def build_cycle_schedule(
    total_days: int = 365,
    dt_days: int = 1,
    inj_days: int = 30,
    shut1_days: int = 5,
    prod_days: int = 30,
    shut2_days: int = 5,
    inj_rate_ton_day: float = 1000.0,
    prod_rate_ton_day: float = 800.0,
    q_ref_ton_day: float | None = None,
) -> Schedule:
    """
    Repeating cycle:
      [inj for inj_days] -> [shut for shut1_days] -> [prod for prod_days] -> [shut for shut2_days] -> repeat

    q_ref_ton_day controls normalization to model units.
    If None, it uses max(inj_rate, prod_rate) so q_model is within [-1, 1].
    """
    total_days = int(max(1, total_days))
    dt_days = int(max(1, dt_days))

    cycle = []
    if inj_days > 0:  cycle += [float(inj_rate_ton_day)] * int(inj_days)
    if shut1_days > 0: cycle += [0.0] * int(shut1_days)
    if prod_days > 0: cycle += [-float(prod_rate_ton_day)] * int(prod_days)
    if shut2_days > 0: cycle += [0.0] * int(shut2_days)
    if not cycle:
        cycle = [0.0]

    nsteps = int(np.ceil(total_days / dt_days))
    q_daily = np.array([cycle[i % len(cycle)] for i in range(total_days)], dtype=np.float32)
    # sample to dt_days
    q = q_daily[::dt_days]
    if q.size < nsteps:
        # pad last value
        q = np.pad(q, (0, nsteps - q.size), mode="edge")
    q = q[:nsteps]

    day = np.arange(nsteps, dtype=np.float32) * float(dt_days)
    t = np.arange(nsteps, dtype=np.int32)

    if q_ref_ton_day is None:
        q_ref_ton_day = float(max(abs(inj_rate_ton_day), abs(prod_rate_ton_day), 1e-9))
    q_model = (q / float(q_ref_ton_day)).astype(np.float32)

    return Schedule(t=t, day=day, q_ton_day=q, q_model=q_model)


def schedule_to_csv_bytes(s: Schedule, use_model_q: bool = True) -> bytes:
    df = pd.DataFrame({
        "t": s.t,
        "day": s.day,
        "q_ton_day": s.q_ton_day,
        "q": s.q_model if use_model_q else s.q_ton_day,
    })
    return df.to_csv(index=False).encode("utf-8")
