"""
HERA-like observation builder.

Wraps `py21cmsense` to produce an (Observatory, Observation, antpos) triple
that can be fed into the `tuesday` noise pipeline.

Requires the optional ``hera`` extra:

    uv sync --extra hera        # in this package
    # or
    uv pip install 'ksz-core[hera]'
"""
from __future__ import annotations

from astropy import units as un
from py21cmsense import GaussianBeam, Observation, Observatory
from py21cmsense.antpos import hera as hera_antpos_fn


def build_hera_observation(
    hex_num: int = 11,
    split_core: bool = True,
    outriggers: int = 2,
    dish_size_m: float = 14.0,
    latitude_deg: float = -30.0,
    track_hours: float = 6.0,
    time_per_day_hours: float = 6.0,
    n_days: int = 180,
):
    """Build a HERA-like Observation. Returns (observatory, observation, antpos)."""
    antpos = hera_antpos_fn(
        hex_num=hex_num, split_core=split_core, outriggers=outriggers,
    )
    observatory = Observatory(
        antpos=antpos,
        latitude=latitude_deg * un.deg,
        beam=GaussianBeam(dish_size=dish_size_m * un.m),
    )
    obs = Observation(
        observatory=observatory,
        track=track_hours * un.hour,
        time_per_day=time_per_day_hours * un.hour,
        n_days=n_days,
    )
    return observatory, obs, antpos
