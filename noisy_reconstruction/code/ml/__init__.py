"""
noisy_reconstruction.code.ml
============================
Box-to-box U-Net pipeline:

    noisy 21cm Tb cube  --(U-Net)-->  line-of-sight velocity v_z cube

All boxes are 21cmFAST coeval cubes at a single redshift chosen so that the
mean neutral fraction is ~0.5 (mid-reionisation). HERA observational noise
is added on top of the 21cm field.

Modules:
    simulate  —  run 21cmFAST coeval cubes and dump (Tb, xHI, v_z) arrays.
    noise     —  HERA uv-sampled thermal noise on the Tb cube.
    dataset   —  torch.utils.data.Dataset over the (noisy_Tb, v_z) pairs.
    unet3d    —  small 3D U-Net (box-to-box, no patches).
    train     —  Adam / MSE training loop with validation + checkpointing.

Defaults (demo):
    HII_DIM = 128           # box cells
    BOX_LEN = 256 [Mpc]     # -> dx = 2 Mpc/cell
    x_HI target ~ 0.5       # mid-reion
    16 train + 4 val sims   # different random seeds, same astro
    HERA-350 layout for the uv-noise
"""

from .config import DemoConfig  # noqa: F401
