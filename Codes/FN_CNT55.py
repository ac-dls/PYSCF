# Here, we try running the heaviest, most nasty and horrible part of the code
# Game: Hybrid DFT (ωB97XV) with RI Coulomb + semi-numerical exchange
# Output: Field Emission Property Profile


import os
import numpy as np
from pyscf import gto, dft, df, scf
from pyscf.scf import _vhf
from pyscf.dft import numint
from types import MethodType
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



# Set up your parameters, dude
xyz_file = "/scratch/dnlrea/pedro.romano/CNT_MET.xyz"
results_dir = "/scratch/dnlrea/pedro.romano/results_S1_CNT55"
os.makedirs(results_dir, exist_ok=True)
basis_large = {'C':'def2-tzvp', 'H':'def2-tzvp', 'O':'def2-tzvp', 'Au':'def2-svpd'}
ecp_large = {'Au':'def2-svpd'}
xc_functional = 'wb97xv'
aux_basis = 'def2-universal-jfit'


ao_file = "/scratch/dnlrea/pedro.romano/ao_vals_memmap_metal.npy"   # <- corrected: values, not grads 
ao_grads_file = "/scratch/dnlrea/pedro.romano/ao_grads_memmap_metal.npy"
ao_grads_memmap = np.lib.format.open_memmap(ao_grads_file, mode='r')
fft_file = "/scratch/dnlrea/pedro.romano/fft_kernel_data_met.npz"
ao_meta = np.load("/scratch/dnlrea/pedro.romano/ao_metadata_met.npz")
grid_coords_file = "/scratch/dnlrea/pedro.romano/grid_coords_met.npz"



AU_EFIELD_V_PER_M = 5.142206747e11   # 1 a.u. electric field in V/m
ANG_TO_AU = 1.0 / 0.52917721092
SAMPLE_SPACING = 3  # every 3rd grid point
vacuum_offset = 12.0  # Å beyond max geometry for vacuum lines
num_vac_lines = 12   # number of sampling lines for V_vac
L_ang = 5.0          #CNT lenght in angstrom
L_m   = L_ang * 1e-10 # conversion to V/m
voltages = np.array([0, 250, 500, 750, 1000], dtype=float)
fn_voltages = voltages[voltages > 0]
E_fields = voltages / L_m
print("Electric fields (V/m):", E_fields)
field_direction = np.array([0.0, 0.0, 0.1])


# Call for building the CNT
mol = gto.Mole(atom=xyz_file, basis=basis_large, ecp=ecp_large, unit='Angstrom')
mol.verbose = 4
mol.max_memory =48000  # MB
mol.build()

nao = mol.nao_nr()  # precomputing dipole integrals once
R_ints = mol.intor('int1e_r', comp=3)   # shape (3, nao, nao)

print("A fresh CNT is out of the oven. Done building it")

# Loading SCF / reference data file templates 
scf_chk_tpl     = "/scratch/dnlrea/pedro.romano/results_S3_CNT55/scf_field_{E}.chk"
mo_info_tpl     = "/scratch/dnlrea/pedro.romano/results_S3_CNT55/MO_info_field{E}.txt"
vacuum_tpl      = "/scratch/dnlrea/pedro.romano/results_S3_CNT55/vacuum_profile_field{E}.txt"
ip_tpl          = "/scratch/dnlrea/pedro.romano/results_S3_CNT55/IP_field{E}.txt"
wf_tpl          = "/scratch/dnlrea/pedro.romano/results_S3_CNT55/work_function_field{E}.txt"


# Load XYZ to compute the identical 'center' used during precompute
atoms_raw = []
with open(xyz_file) as f:
    lines = f.readlines()[2:]
    for line in lines:
        parts = line.split()
        atoms_raw.append(np.array(parts[1:4], float))
coords_raw = np.array(atoms_raw)
center = coords_raw.mean(axis=0) * ANG_TO_AU  # convert to Bohr

# Shift R integrals so that dipole is referenced to the centered geometry
R_ints_shifted = R_ints - center[:, None, None]

# Ensure base_hcore exists before we use it (nuclear + kinetic + ECP terms)
# (safe: create a temporary MF and get its hcore)
_tmp_mf = dft.RKS(mol)
base_hcore = _tmp_mf.get_hcore()   # matrix in atomic-orbital basis (Bohr units)
del _tmp_mf


# Setting up the fields for SCF computation 
for volt, E in zip(fn_voltages, E_fields[voltages > 0]):
    
    E_int = int(volt)   # E is already in V/m
    E_au = float(E) / AU_EFIELD_V_PER_M
    n = field_direction / np.linalg.norm(field_direction)

    # 1. Loading SCF solution
    chkfile = scf_chk_tpl.format(E=E_int)
    if not os.path.exists(chkfile):
        raise FileNotFoundError(f"Missing SCF checkpoint: {chkfile}")

    mf = dft.RKS(mol, xc=xc_functional)
    mf.chkfile = chkfile

    # Load SCF data explicitly (PySCF 2.1.1 compatible)
    scf_data = scf.chkfile.load(chkfile, "scf")

    mf.mo_coeff = scf_data["mo_coeff"]
    mf.mo_energy = scf_data["mo_energy"]
    mf.mo_occ = scf_data["mo_occ"]
    mf.e_tot = scf_data["e_tot"]

    # Build density matrix explicitly
    dm = mf.make_rdm1()

    assert hasattr(mf, "make_rdm1")
    assert dm.shape == (nao, nao)


    
    # 2. MO energies / HOMO 
    mo_energy = mf.mo_energy
    mo_occ    = mf.mo_occ

    homo_idx = np.where(mo_occ > 0)[0].max()
    homo_energy = mo_energy[homo_idx]   # Hartree

    print(f"HOMO energy (Ha): {homo_energy:.12f}")

    # 3. Load vacuum level
    vacuum_file = vacuum_tpl.format(E=E_int)
    if not os.path.exists(vacuum_file):
        raise FileNotFoundError(f"Missing vacuum file: {vacuum_file}")

    vac_data = np.loadtxt(vacuum_file)

    # Last column is V_loc(Ha)
    n_tail = max(1, int(0.2 * len(vac_data)))
    v_vac = float(np.mean(vac_data[-n_tail:, -1]))


    print(f"Vacuum level (Ha): {v_vac:.12f}")

    # 4. Load ionization potential 
    ip_file = ip_tpl.format(E=E_int)
    if not os.path.exists(ip_file):
        raise FileNotFoundError(f"Missing IP file: {ip_file}")

    IP = None

    with open(ip_file, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "IP_Ha":
                IP = float(parts[1])
                break

    if IP is None:
        raise RuntimeError(f"IP_Ha not found in {ip_file}")

    print(f"Ionization potential (Ha): {IP:.12f}")
    print(f"Loaded IP = {IP:.6f} Ha ({IP*27.2114:.3f} eV)")


    # 5. Load work function
    wf_file = wf_tpl.format(E=E_int)
    if not os.path.exists(wf_file):
        raise FileNotFoundError(f"Missing WF file: {wf_file}")

    WF = None

    with open(wf_file, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Case 1: labeled format (WF_Ha 0.163...)
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0] == "WF_Ha":
            WF = float(parts[1])
            break

    # Case 2: numeric-only format (first numeric line = WF_Ha)
    if WF is None:
        try:
            WF = float(lines[0])
        except Exception:
            raise RuntimeError(f"Could not parse WF from {wf_file}")

    print(f"Work function loaded: {WF:.12f} Ha ({WF*27.2114:.6f} eV)")
        
    print(f"Loaded WF = {WF:.6f} Ha ({WF*27.2114:.3f} eV)")

    # 6. Energy consistency sanity check 
    err = abs(v_vac - homo_energy - IP)

    if err > 5e-3:
        print(
            "WARNING: Energy alignment inconsistency:\n"
            f"  vacuum - HOMO = {v_vac - homo_energy:.6f} Ha\n"
            f"  IP            = {IP:.6f} Ha\n"
            f"  |Δ|           = {err:.3e} Ha"
        )

   
    # 7. Loading AO MEMMAP precomputed data 

    print("Loading AO grid")
    ao_memmap = np.lib.format.open_memmap(ao_file, mode='r')

    if ao_memmap.ndim != 2:
            raise RuntimeError(f"AO memmap must be 2D (ngrids, nao). Got {ao_memmap.shape}")

    ngrids_ao, nao_ao = ao_memmap.shape
    meta_nao = int(ao_meta["nao"])
    meta_npts = int(ao_meta["npts"])
    
    if nao_ao != meta_nao:
        raise RuntimeError(f"AO memmap nao ({nao_ao}) != ao_metadata nao ({meta_nao})")
    
    if ngrids_ao != meta_npts:
        raise RuntimeError(f"AO memmap npts ({ngrids_ao}) != ao_metadata npts ({meta_npts})")
    
    if nao_ao != nao:
        raise RuntimeError(
            f"AO mismatch: memmap has nao={nao_ao}, molecule has nao={nao}"
        )

    if ao_grads_memmap.shape != (3, ngrids_ao, nao):
        raise RuntimeError("AO gradients memmap has wrong shape")

   
    # 8. Loading FFT metadata 
    fft_data = np.load(fft_file)
    dx_au = float(fft_data['dx_au'])
    shape_pad = np.array(fft_data['shape_pad'], dtype=int)
    origin_pad_bohr = np.asarray(fft_data['origin_pad_bohr'], dtype=float)
    kernel_ft_pre = fft_data['kernel_ft'].astype(np.complex128)
    

    # Sanity prints
    print("FFT kernel metadata loaded:")
    print("  shape_pad =", shape_pad)
    print("  dx_au =", dx_au)
    print("  origin_pad_bohr =", origin_pad_bohr)

    
    # 4. padded axes in Bohr (used for interpolator)
    Nx_pad, Ny_pad, Nz_pad = tuple(shape_pad)
    xs_pad = origin_pad_bohr[0] + np.arange(Nx_pad) * dx_au
    ys_pad = origin_pad_bohr[1] + np.arange(Ny_pad) * dx_au
    zs_pad = origin_pad_bohr[2] + np.arange(Nz_pad) * dx_au

    # 9. Loading the small-grid metadata

    if not os.path.exists(grid_coords_file):
        raise FileNotFoundError(f"{grid_coords_file}")
    
    grid_data = np.load(grid_coords_file)
    keys = grid_data.files

    grid_pts_ang = grid_data["grid_coords_ang"]
    if "grid_coords_ang" in keys:
        grid_pts_ang = grid_data["grid_coords_ang"]
    else: 
        raise RuntimeError("grid_coords_met must contain grid_coords_ang")
    
    
    grid_pts_bohr = grid_pts_ang * ANG_TO_AU
    
    if grid_pts_bohr.shape[0] != ngrids_ao:
        raise ValueError(
            f"Grid size mismatch, dude: coords={grid_pts_bohr.shape[0]}, AOmem={ngrids_ao}"
        )
    
   
    print("Available keys in grid_coords_met.npz:", keys)

    
    # 10. Loading small grid origin 

    if "origin_small_bohr" in keys:
        origin_small_bohr = grid_data["origin_small_bohr"].astype(float)
    elif "origin_small_ang" in keys:
        origin_small_bohr = grid_data["origin_small_ang"].astype(float) * ANG_TO_AU
    else: 
        raise RuntimeError("origin_small_bohr missing in grid metadata")
    
    # 11. Loading pad_left

    if "pad_left" in grid_data.files:
        pad_left = np.asarray(grid_data["pad_left"], dtype=int)
    else:
        pad_left = np.round((origin_small_bohr - origin_pad_bohr) / dx_au).astype(int)
    
    # 12. Loading small grid shape (nx, ny, nz)

    if {"nx", "ny", "nz"} <= set(keys):
        nx = int(grid_data["nx"])
        ny = int(grid_data["ny"])
        nz = int(grid_data["nz"])
        shape_small = np.array([nx, ny, nz], int)
    elif "shape_small" in keys: 
        shape_small = np.asarray(grid_data["shape_small"], int)
        nx, ny, nz = shape_small
    else: 
            raise RuntimeError("small-grid shape is missing, dude")
    
    print("Loaded small grid metadata:")
    print("  origin_small_bohr =", origin_small_bohr)
    print("  pad_left =", pad_left)
    print("  shape_small =", shape_small)


    # 13. Critical Check: ensure small-grid origin matches the padded-grid origin + pad_left
    
    origin_small_from_pad = origin_pad_bohr + pad_left * dx_au
    diff = np.max(np.abs(origin_small_from_pad - origin_small_bohr))
    if diff > 0.5 * dx_au:
            raise RuntimeError(
                f"Inconsistent origins!\n"
                f"origin_small_bohr = {origin_small_bohr}\n"
                f"origin_from_pad   = {origin_small_from_pad}\n"
                f"diff = {diff}"
            )

    
    # 14. External Uniform-grid field operator (for dipole)
    Rdot = n[0] * R_ints_shifted[0] + n[1] * R_ints_shifted[1] + n[2] * R_ints_shifted[2]
    h_field = - E_au * Rdot   # electrons feel -E·r

    # 15. Density reconstruction on small grid 
    print("Reconstructing electron density on AO grid")

    ao_vals = ao_memmap  # (ngrids, nao)

    rho_small = np.einsum(
        "gm,mn,gn->g",
        ao_vals,
        dm,
        ao_vals,
        optimize=True
    )

    rho_small = rho_small.reshape(shape_small)

    # Electron count sanity check
    nelec_grid = rho_small.sum() * (dx_au ** 3)
    print(f"Integrated electron count (grid): {nelec_grid:.6f}")

    # 16. Embed density into padded FFT box 
    rho_pad = np.zeros(shape_pad, dtype=np.float64)

    ix, iy, iz = pad_left
    nx, ny, nz = shape_small

    rho_pad[
        ix:ix+nx,
        iy:iy+ny,
        iz:iz+nz
    ] = rho_small

    # 17. Hartree potential via FFT ===
    print("Computing Hartree potential via FFT")

    rho_ft = np.fft.fftn(rho_pad)
    vh_ft = rho_ft * kernel_ft_pre
    vh_pad = np.fft.ifftn(vh_ft).real

    # 18. Extract small-grid Hartree potential
    V_H_small = vh_pad[
        ix:ix+nx,
        iy:iy+ny,
        iz:iz+nz
    ]
    
    # 19. Nuclear potential 
    print("Computing nuclear potential")

    print("Computing nuclear potential")

    coords_bohr = mol.atom_coords()   # (Natoms, 3)
    charges = mol.atom_charges()

    grid_flat = grid_pts_bohr.reshape((-1, 3))  # (Ngrid, 3)

    V_nuc_flat = np.zeros(grid_flat.shape[0], dtype=np.float64)

    for Z, R in zip(charges, coords_bohr):
        r = np.linalg.norm(grid_flat - R[None, :], axis=1)
        V_nuc_flat -= Z / np.where(r > 1e-8, r, np.inf)

    V_nuc = V_nuc_flat.reshape(shape_small)

    print("V_nuc sanity:")
    print("  min =", V_nuc.min())
    print("  max =", V_nuc.max())


    # 20. External field potential 
    x_coords = grid_flat[:, 0].reshape(shape_small)
    V_ext = -E_au * x_coords
    
    # 21. Electrostatic Emission Potential 
    V_total = V_H_small + V_nuc + V_ext

    # 22. Local Electric Field
    print("Computing local electric field")
    Ex, Ey, Ez = np.gradient(
        V_total,
        dx_au,
        dx_au,
        dx_au,
        edge_order=2
    )

    # Electric field = -∇V
    Ex = -Ex
    Ey = -Ey
    Ez = -Ez

    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    
    # 23. Field Enhancement Factor
    E_applied = abs(E_au)

    rho_thresh = 1e-6
    vac_mask = rho_small < rho_thresh
    E_local_max = np.max(E_mag[vac_mask])

    if E_applied > 1e-10:
        beta = E_local_max / E_applied
        print(f"Field enhancement β = {beta:.2f}")

        np.savetxt(
            os.path.join(results_dir, f"beta_field_{E_int}.txt"),
            [beta]
        )
    else:
        beta = np.nan
        print("E_applied = 0 → β undefined (skipping)")
    
    # 24. 1D Emission Barrier along x (average over y,z)
    V_barrier = V_total.mean(axis=(1,2))
    x_axis = xs_pad[ix:ix+nx]  # Bohr

    np.savetxt(
        os.path.join(results_dir, f"barrier_field_{E_int}.txt"),
        np.column_stack((x_axis, V_barrier)),
        header="x (Bohr)   V_total (Ha)"
    )
    
    # 25. Fowler–Nordheim emission current density
    if E_applied > 1e-10:
        A_FN = 1.54e-6      # A eV V^-2
        B_FN = 6.83e9       # eV^-3/2 V/m
        phi_eV = WF * 27.2114
        F_local = E_local_max * AU_EFIELD_V_PER_M  # V/m

        J = (A_FN * F_local**2 / phi_eV) * np.exp(
            -B_FN * phi_eV**1.5 / F_local
        )

        sigma_eff = J / (E_applied * AU_EFIELD_V_PER_M)

        print(f"Emission current density J = {J:.3e} A/m^2")
        print(f"Effective conductivity σ = {sigma_eff:.3e} S/m")

    else:
        J = 0.0
        sigma_eff = np.nan
        print("E_applied = 0 → FN current and conductivity undefined")

    np.savetxt(
        os.path.join(results_dir, f"emission_current_{E_int}.txt"),
        [J]
    )

    np.savetxt(
        os.path.join(results_dir, f"conductivity_{E_int}.txt"),
        [sigma_eff]
    )


    # 26. Effective Conductivity
    sigma_eff = J / (E_applied * AU_EFIELD_V_PER_M)

    np.savetxt(
        os.path.join(results_dir, f"conductivity_{E_int}.txt"),
        [sigma_eff]
    )

# Proper scientific notation format 
def sci_fmt(val, pos):
    if val == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / 10**exponent
    return r"${:.2f}\times 10^{{{:d}}}$".format(mantissa, exponent)

plt.figure()
plt.plot(x_axis * 0.529177, V_barrier * 27.2114)
plt.xlabel("x (Å)")
plt.ylabel("Potential (eV)")
plt.title(f"Emission Barrier ({E_int} V)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"barrier_{E_int}.png"), dpi=300)
plt.close()

plt.figure()
plt.hist(E_mag[vac_mask] / E_applied, bins=100)
plt.xlabel("β")
plt.ylabel("Counts")
plt.title(f"Field Enhancement Distribution ({E_int} V)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"beta_distribution_{E_int}.png"), dpi=300)
plt.close()

print("Done, dude!")

        

