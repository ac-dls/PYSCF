# ----------------------------------------------------------
# Minimal Precompute Script for Step 1 (WF / IP / charges)
# ----------------------------------------------------------
import numpy as np
from pyscf import gto

xyz_file = "/scratch/dnlrea/pedro.romano/CNT_CHIRAL_AU.xyz"

basis_large = {'C':'def2-tzvp', 'H':'def2-tzvp', 'O':'def2-tzvp', 'Au':'def2-tzvp'}
ecp_large   = {'Au':'def2-tzvp'}   # only Au uses ECP

# 1. Load XYZ and center geometry
atoms = []
coords_raw = []

with open(xyz_file) as f:
    lines = f.readlines()[2:]  # skip "N" and comment line

for line in lines:
    parts = line.split()
    sym = parts[0]
    xyz = np.array(list(map(float, parts[1:4])))
    atoms.append(sym)
    coords_raw.append(xyz)

coords_raw = np.array(coords_raw)

# Compute centroid in Å
center = coords_raw.mean(axis=0)
coords_centered = coords_raw - center

# 2. Build PySCF molecule with centered geometry
atom_str = ""
for sym, c in zip(atoms, coords_centered):
    atom_str += f"{sym} {c[0]} {c[1]} {c[2]}; "

mol = gto.Mole(
    atom = atom_str,
    basis = basis_large,
    ecp   = ecp_large,
    unit  = 'Angstrom'
)
mol.build()

# 3. Save the shift (center) used – SCF will need this
np.savez("precompute_geom_au.npz",
         center_used_ang=center,
         coords_centered_ang=coords_centered,
         atom_symbols=np.array(atoms))

print("Precompute complete: geometry centered correctly.")
print("Center shift (Å):", center)