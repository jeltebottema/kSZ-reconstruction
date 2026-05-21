"""
simulate.py
===========
Generate 21cmFAST coeval cubes at a single redshift where <x_HI> ~ 0.5
(mid-reionisation) and dump

    (Tb, Tb_real, xHI, vz)  -> data/cubes/sim_seed<N>.npz

Targets py21cmfast v4 (4.1.1). The saved `Tb` field is the
redshift-space brightness temperature (RSDs applied via
`py21cmfast.rsds.apply_rsds`), since that is what carries vz
information for the downstream U-Net. `Tb_real` is the real-space
version, kept for diagnostics. `vz` is saved in km/s (converted from
the Mpc/s unit used internally by py21cmfast v4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .config import DemoConfig

# Mpc -> km
MPC_PER_KM = 3.0857e19


def _build_inputs(cfg: DemoConfig, z: float, seed: int):
    """Assemble a py21cmfast v4 InputParameters for (z, seed)."""
    import py21cmfast as p21c

    simulation_options = p21c.SimulationOptions(
        HII_DIM=cfg.hii_dim, BOX_LEN=cfg.box_len,
    )
    matter_options = p21c.MatterOptions(
        SOURCE_MODEL="CONST-ION-EFF",   # halo-free; no HaloBox required
        KEEP_3D_VELOCITIES=True,        # needed for apply_rsds
    )
    astro_options = p21c.AstroOptions(
        USE_TS_FLUCT=False,
        USE_EXP_FILTER=False,            # incompatible with CONST-ION-EFF
        USE_UPPER_STELLAR_TURNOVER=False,
    )
    return p21c.InputParameters(
        simulation_options=simulation_options,
        matter_options=matter_options,
        cosmo_params=p21c.CosmoParams(),
        astro_params=p21c.AstroParams(),
        astro_options=astro_options,
        random_seed=seed,
        node_redshifts=[z],
    )


def _run_coeval(cfg: DemoConfig, z: float, seed: int):
    """Run IC -> PerturbedField -> IonizedBox -> BrightnessTemp and return them."""
    import py21cmfast as p21c

    inputs = _build_inputs(cfg, z, seed)
    ic = p21c.compute_initial_conditions(inputs=inputs)
    pf = p21c.perturb_field(redshift=z, inputs=inputs, initial_conditions=ic)
    ib = p21c.compute_ionization_field(
        perturbed_field=pf, initial_conditions=ic, inputs=inputs,
    )
    bt = p21c.brightness_temperature(ionized_box=ib, perturbed_field=pf)
    return inputs, ic, pf, ib, bt


def _scan_for_xhi(cfg: DemoConfig) -> float:
    """Scan cfg.z_grid with seed0 to find z at cfg.target_xHI."""
    mean_xhi: dict[float, float] = {}
    for z in cfg.z_grid:
        _, _, _, ib, _ = _run_coeval(cfg, z, cfg.seed0)
        mean_xhi[z] = float(ib.neutral_fraction.value.mean())
        print(f"  scan: z={z:5.2f}  <x_HI>={mean_xhi[z]:.3f}")

    zs = np.array(sorted(mean_xhi))
    xs = np.array([mean_xhi[z] for z in zs])
    order = np.argsort(xs)
    z_target = float(np.interp(cfg.target_xHI, xs[order], zs[order]))
    print(f"  -> z(<x_HI>={cfg.target_xHI}) ~ {z_target:.3f}")
    return z_target


def generate_dataset(cfg: DemoConfig, z_fixed: float | None = None) -> Dict[str, Path]:
    """Run 21cmFAST for (n_train + n_val) seeds, apply RSDs, save cubes.

    Returns {"train": [paths], "val": [paths], "z": z_used}.
    """
    from py21cmfast.rsds import apply_rsds

    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    if z_fixed is None:
        print("Scanning for z at target x_HI ...")
        z_fixed = _scan_for_xhi(cfg)

    train_paths, val_paths = [], []

    for i in range(cfg.n_total):
        seed = cfg.seed0 + i
        out = cfg.data_dir / f"sim_seed{seed}.npz"
        tag = "train" if i < cfg.n_train else "val"
        bucket = train_paths if tag == "train" else val_paths

        if out.exists():
            print(f"[{tag:5s}] seed={seed} already exists, skipping")
            bucket.append(out)
            continue

        print(f"[{tag:5s}] seed={seed}  running 21cmFAST coeval at z={z_fixed:.3f} ...")
        inputs, _, pf, ib, bt = _run_coeval(cfg, z_fixed, seed)

        vz_mpcs = pf.velocity_z.value.astype(np.float64)
        tb_real = bt.brightness_temp.value.astype(np.float64)
        xHI     = ib.neutral_fraction.value.astype(np.float32)

        tb_rsd = np.asarray(
            apply_rsds(field=tb_real, los_velocity=vz_mpcs,
                       redshifts=z_fixed, inputs=inputs, periodic=True),
            dtype=np.float32,
        )
        tb_real = tb_real.astype(np.float32)
        vz_kms  = (vz_mpcs * MPC_PER_KM).astype(np.float32)

        assert tb_rsd.shape == (cfg.hii_dim,) * 3, f"unexpected Tb shape {tb_rsd.shape}"
        assert vz_kms.shape == tb_rsd.shape,       f"vz shape mismatch {vz_kms.shape}"

        np.savez_compressed(
            out,
            Tb=tb_rsd,          # redshift-space Tb — the training input
            Tb_real=tb_real,    # real-space Tb — diagnostic only
            xHI=xHI,
            vz=vz_kms,          # km/s
            z=z_fixed, seed=seed,
            hii_dim=cfg.hii_dim, box_len=cfg.box_len,
        )
        bucket.append(out)
        print(f"        -> {out.name}  <x_HI>={xHI.mean():.3f}  "
              f"Tb_RSD std={tb_rsd.std():.2f} mK  vz std={vz_kms.std():.1f} km/s")

    return {"train": train_paths, "val": val_paths, "z": z_fixed}


if __name__ == "__main__":
    generate_dataset(DemoConfig())
