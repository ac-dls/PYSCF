# Precomputing (CNT_MET = 5 angstrom in lenght)
#  Here we center the CNT .xyz, we generate and align the AO and FFT grids so that they can overlap properly when we run the scf cycles
import os
import numpy as np
from pyscf import gto
from pyscf.dft import numint

# Simualtion parameters: Show me the baby
xyz_file = "/scratch/dnlrea/pedro.romano/CNT_CHIRAL.xyz"

basis_large = {'C':'def2-tzvp', 'H':'def2-tzvp', 'O':'def2-tzvp', 'Au':'def2-tzvp'}
ecp_large   = {'Au':'def2-tzvp'}

pad_axial  = 8.0  
pad_radial = 8.0    
dx = 0.40           # grid spacing [Å]
ANG_TO_AU = 1.0 / 0.52917721092
# convert grid spacing once, use consistently
dx_au = dx * ANG_TO_AU   # [Bohr]

# 1. Loading the .XYZ safely and recentering the geometry
# Read atoms manually and we want the coords
atoms = []
with open(xyz_file) as f:
    lines = f.readlines()[2:]

for line in lines:
    parts = line.split()
    atoms.append((parts[0], tuple(map(float, parts[1:4]))))

# Converting to array
coords_raw = np.array([c for _, c in atoms])

# Recentering the geometry of the CNT
# Worth mentioning that this is critical for avoiding offset gridsand future asymmetric potentials)
center = coords_raw.mean(axis=0)
coords_centered = coords_raw - center

# Constructing the atom string for PySCF
atom_str = ""
for (sym, _), c in zip(atoms, coords_centered):
    atom_str += f"{sym} {c[0]} {c[1]} {c[2]}; "

mol_large = gto.Mole(
    atom = atom_str,
    basis = basis_large,
    ecp   = ecp_large,
    unit  = 'Angstrom'
)
mol_large.build()

coords = mol_large.atom_coords(unit='Angstrom')

# 2. Determine bounding box with custom axial/radial pads
min_xyz = coords.min(0)
max_xyz = coords.max(0)
spreads = max_xyz - min_xyz

axis = int(np.argmax(spreads))   # nanotube axis

pad = np.array([pad_radial, pad_radial, pad_radial])
pad[axis] = pad_axial   # axial pad only in long direction

# These will be the extremes of the grid, love
min_xyz -= pad
max_xyz += pad


# 3. Building an uniform grid
xs = np.arange(min_xyz[0], max_xyz[0] + 1e-12, dx)
ys = np.arange(min_xyz[1], max_xyz[1] + 1e-12, dx)
zs = np.arange(min_xyz[2], max_xyz[2] + 1e-12, dx)

X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

grid_pts_ang  = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
grid_pts_bohr = grid_pts_ang * ANG_TO_AU
nx, ny, nz = len(xs), len(ys), len(zs)
assert grid_pts_bohr.shape == (nx*ny*nz, 3)


# Saving the info to avoid confusion later on
npts_expected = nx * ny * nz

grid_metadata_extra = {
    "nx": nx,
    "ny": ny,
    "nz": nz,
    "npts_expected": npts_expected,
    "grid_shape_small": np.array([nx, ny, nz], dtype=int),
    "dx_ang": dx
}

# Current (un-padded) grid origin in Angstrom:
origin_ang = np.array([xs[0], ys[0], zs[0]])        # equals min_xyz
origin_bohr = origin_ang * ANG_TO_AU  #Origin in bohr real-space

# Number of points in unpadded grid
shape = np.array([nx, ny, nz], dtype=int)

# Padded shape (integers)
pad_factor = 1.5
shape_pad = np.array([int(np.ceil(s * pad_factor)) for s in shape], dtype=int)

# Ensuring the padded shape is >= original shape
shape_pad = np.maximum(shape_pad, shape)

# How many padding points total along each axis
pad_tot = shape_pad - shape

# Here, I'm centering the original grid inside the padded box (half pad on left, half on right)
pad_left = (pad_tot // 2).astype(int)
pad_right = (pad_tot - pad_left).astype(int)

shape_small = np.array([nx, ny, nz], dtype=int)

# The padded grid origin (in Bohr) is shifted by -pad_left * dx_au relative to origin_bohr
dx_au = dx * ANG_TO_AU

origin_small_ang = origin_ang.copy()
origin_small_bohr = origin_bohr.copy()


origin_pad_bohr = origin_bohr - pad_left * dx_au
origin_pad_ang = origin_ang - pad_left * dx

slice_small = np.array([
    pad_left,
    pad_left + shape_small
])

# Saving the grid metadata --> This is the single source of truth, babe
# Saving the grid metadata --> This is the single source of truth for grid/FFT geometry
np.savez("grid_coords_chiral.npz",
         grid_coords_ang=grid_pts_ang,
         grid_coords_bohr=grid_pts_bohr,
         origin_ang=origin_ang,
         origin_bohr=origin_bohr,
         origin_small_ang=origin_small_ang,
         origin_small_bohr=origin_small_bohr,
         origin_pad_ang=origin_pad_ang,
         origin_pad_bohr=origin_pad_bohr,
         nx=nx, ny=ny, nz=nz,
         shape_small=shape_small,
         shape_pad=shape_pad,
         pad_left=pad_left,
         pad_right=pad_right,
         slice_small=slice_small,
         dx_au=dx_au,
         dx_ang=dx)


# 4. AO Evaluation and memmap formation
nao = mol_large.nao_nr()
npts = len(grid_pts_bohr)
ao_vals_file  = "ao_vals_memmap_chiral.npy"         # (N, nao)  - existing name, reused
ao_grads_file = "ao_grads_memmap_chiral.npy"       # (3, N, nao) gradients w.r.t x,y,z in BOHR units


# create memmaps (float64)
ao_vals = np.lib.format.open_memmap(ao_vals_file,
                                    mode='w+',
                                    dtype=np.float64,
                                    shape=(npts, nao))

# store grads in shape (3, npts, nao)
ao_grads = np.lib.format.open_memmap(ao_grads_file,
                                     mode='w+',
                                     dtype=np.float64,
                                     shape=(3, npts, nao))

# Saving AO memmaps for SCF consistency AFTER memmaps are created
ao_vals_shape = ao_vals.shape
ao_grads_shape = ao_grads.shape

np.savez("ao_metadata_chiral.npz",
         ao_vals_shape=ao_vals_shape,
         ao_grads_shape=ao_grads_shape,
         nao=nao,
         npts=npts,
         nx=nx, ny=ny, nz=nz,
         ao_labels=mol_large.ao_labels(),
         file_vals=ao_vals_file,
         file_grads=ao_grads_file,
         dx_au=dx_au,
         dx_ang=dx)

print("Saved AO memmap metadata:")
print("  ao_vals_shape =", ao_vals_shape)
print("  ao_grads_shape =", ao_grads_shape)
print("ao_vals sanity:")
print("  min =", ao_vals.min())
print("  max =", ao_vals.max())

batch_size = 20000
print(f"Evaluating AO basis and gradients on grid ({nao} AOs)…")

for start in range(0, npts, batch_size):
    end = min(start + batch_size, npts)
    coords_chunk = grid_pts_bohr[start:end]  # Bohr

    # AO values
    ao_val_chunk = mol_large.eval_gto("GTOval", coords_chunk)

    # AO gradients
    ao_grad_chunk = mol_large.eval_gto("GTOval_ip", coords_chunk)

    # Store values
    ao_vals[start:end, :] = ao_val_chunk

    # Store gradients
    ao_grads[0, start:end, :] = ao_grad_chunk[0]
    ao_grads[1, start:end, :] = ao_grad_chunk[1]
    ao_grads[2, start:end, :] = ao_grad_chunk[2]

    print(f"AO chunk {start}:{end} done")
    print("Post-AO sanity:")
    print("  AO min =", ao_vals.min())
    print("  AO max =", ao_vals.max())

    if ao_vals.max() < 1e-8:
        raise RuntimeError("AO values are numerically zero — AO evaluation failed")


# flush and delete memmaps
ao_vals.flush()
ao_grads.flush()
del ao_vals, ao_grads
print("AO values + gradients saved to disk.")

# FINAL SAFETY CHECK: re-open read-only and confirm size matches expectation
ao_chk = np.lib.format.open_memmap(ao_vals_file, mode='r')
if ao_chk.shape[0] != npts_expected:
    raise RuntimeError(
        f"AO grid size mismatch: AO memmap has {ao_chk.shape[0]} points "
        f"but expected {npts_expected} = nx*ny*nz"
    )
print("AO consistency check passed.")
del ao_chk


# FINAL SAFETY CHECK
ao = np.lib.format.open_memmap(ao_vals_file, mode='r')
if ao.shape[0] != npts_expected:
    raise RuntimeError(
        f"AO grid size mismatch: AO memmap has {ao.shape[0]} points "
        f"but expected {npts_expected} = nx*ny*nz"
    )
print("AO consistency check passed.")


# 5. FFT Kernel with padding 
dx_au = dx * ANG_TO_AU

kx = np.fft.fftfreq(shape_pad[0], d=dx_au)
ky = np.fft.fftfreq(shape_pad[1], d=dx_au)
kz = np.fft.fftfreq(shape_pad[2], d=dx_au)

KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

kernel_ft = np.zeros_like(K2)
mask = (K2 != 0.0)
kernel_ft[mask] = 4 * np.pi / K2[mask]


# Also saving dx in Angstrom and molecule electron count
dx_ang = dx
mol_nelec = mol_large.nelectron

np.savez(
    "fft_kernel_data_chiral.npz",
    K2=K2,
    kernel_ft=kernel_ft,
    shape_pad=shape_pad,
    dx_au=dx_au,
    dx_ang=dx_ang,
    origin_pad_bohr=origin_pad_bohr,
    pad_left=pad_left,
    shape_small=shape_small,
    origin_small_bohr=origin_small_bohr,
    mol_nelec=mol_nelec
)

# Sanity checks (fail early if we have mismatches)
ao = np.lib.format.open_memmap(ao_vals_file, mode='r')
npts_ao = ao.shape[0]
expected = int(nx) * int(ny) * int(nz)
assert npts_ao == expected, (
    f"AO memmap points ({npts_ao}) != expected grid points ({expected}). "
    "Check grid construction or AO memmap writing."
)
print("Precompute sanity check OK: AO memmap points == nx*ny*nz =", expected)
print("Saved grid metadata keys:", np.load("grid_coords_chiral.npz").files)
print("Saved fft metadata keys:", np.load("fft_kernel_data_chiral.npz").files)

# small unit test: check ao_metadata_met.npz can be loaded and shapes make sense
meta = np.load("ao_metadata_chiral.npz")
ao_vals_chk = np.lib.format.open_memmap(ao_vals_file, mode='r')
ao_grads_chk = np.lib.format.open_memmap(ao_grads_file, mode='r')

assert ao_vals_chk.shape == tuple(meta["ao_vals_shape"])
assert ao_grads_chk.shape == tuple(meta["ao_grads_shape"])



print(f"AO file size: {os.path.getsize(ao_vals_file)/1e9:.2f} GB")
print("Precomputation complete!")
print("Done, dude!")
