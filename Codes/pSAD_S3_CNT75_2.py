# Script 3: Field Emission Potential and Stabilization Energies under E-field for CNT (5,5)

# Trying the SCF convergence using pSAD strategy to help generate the electrical properties of the CNT

"""
SCF helper script with layered initial-guess strategy:
  SAD -> purified-SAD (pSAD) -> SAP-like (FD on H_core) -> MINAO

Supports:
- Explicit finite-T SCF (PySCF 2.7.0: fermi smearing not supported internally).
- Field loop with checkpoint reuse.
- Heatmaps with AO labels and colored axis ticks.
"""

import os, sys, math
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from pyscf import gto, scf
from pyscf import dft
from pyscf import lib
from pyscf.tools import cubegen
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
import shutil
from pyscf import df



# Loading the precomputed aligned system info
geom = np.load("/scratch/dnlrea/pedro.romano/precompute_geom_chiral.npz")
coords_ang = geom["coords_centered_ang"]   # (Nat, 3)
symbols    = geom["atom_symbols"]          # list/array of strings

atoms = [(sym, tuple(coord)) for sym, coord in zip(symbols, coords_ang)]

# Settings for the CNT system
xyz_file = "/scratch/dnlrea/pedro.romano/CNT_CHIRAL.xyz"
results_dir = "results_S3_CNT75"
basis_set = "def2-tzvp"
xc_functional = "wb97xv"
spin = 0
charge = 0
max_scf_cycles = 500        # increased from 200
conv_tol = 1e-6             # tighter tolerance for final PBE stage
max_memory_mb = 8000

purify_max_iter = 30
purify_tol = 1e-8
use_fd_smeared_guess = True
kT_ha = 0.001
cube_nxyz = (60, 60, 60)

# Unified field array: includes 0 (reference) and thruster range 
L_ang = 5.0          #CNT lenght in angstrom
L_m   = L_ang * 1e-10 # conversion to V/m
voltages = np.array([0, 250, 500, 750, 1000], dtype=float)
E_fields = voltages / L_m
print("Electric fields (V/m):", E_fields)
field_direction = np.array([1.0, 0.0, 0.0])
AU_EFIELD_V_PER_M = 5.142206747e11   # 1 a.u. electric field in V/m
ANG_TO_AU = 1.0 / 0.52917721092
V_PER_M_TO_AU = 1.0 / 5.142206747e11

os.makedirs(results_dir, exist_ok=True)

# Helper functions

def save_np_txt(name, arr):
    np.savetxt(os.path.join(results_dir, name), arr)

def is_finite_matrix(M):
    return (isinstance(M, np.ndarray) and np.all(np.isfinite(M)))

def safe_symmetrize_and_normalize(D, S=None, nocc=None):
    if D is None:
        return None
    D = 0.5 * (D + D.T)
    if S is not None and nocc is not None:
        try:
            tr = np.trace(D @ S)
            if tr != 0 and np.isfinite(tr):
                D = D * (float(nocc) / float(tr))
        except Exception:
            pass
    return D

def mcweeny_purify(D, S=None, nocc=None, max_iter=30, tol=1e-8):
    if D is None:
        return None, 0
    D = np.array(D, dtype=float, copy=True)
    D = 0.5 * (D + D.T)
    for it in range(1, max_iter + 1):
        if not np.all(np.isfinite(D)):
            return None, it-1
        try:
            D2 = D @ D
        except Exception:
            return None, it-1
        err = np.linalg.norm(D2 - D)
        if err < tol:
            D = 0.5 * (D + D.T)
            if S is not None and nocc is not None:
                try:
                    tr = np.trace(D @ S)
                    if tr != 0 and np.isfinite(tr):
                        D *= (float(nocc) / float(tr))
                except Exception:
                    pass
            return D, it
        try:
            D_new = 3.0 * D2 - 2.0 * (D2 @ D)
        except Exception:
            return None, it-1
        D_new = 0.5 * (D_new + D_new.T)
        if S is not None and nocc is not None:
            try:
                tr = np.trace(D_new @ S)
                if tr != 0 and np.isfinite(tr):
                    D_new *= (float(nocc) / float(tr))
            except Exception:
                pass
        if not np.all(np.isfinite(D_new)):
            return None, it
        D = D_new
    return D, max_iter

def build_fd_density_from_HS(H, S, nocc, kT):
    eps, C = la.eigh(H, S)
    emin, emax = float(eps.min() - 10 * kT), float(eps.max() + 10 * kT)
    def total_occ(mu):
        f = 1.0 / (1.0 + np.exp((eps - mu) / kT))
        return f.sum()
    mu_lo, mu_hi = emin, emax
    for _ in range(200):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        if total_occ(mu_mid) > nocc:
            mu_hi = mu_mid
        else:
            mu_lo = mu_mid
        if abs(mu_hi - mu_lo) < 1e-10:
            break
    mu = 0.5 * (mu_lo + mu_hi)
    f = 1.0 / (1.0 + np.exp((eps - mu) / kT))
    D = (C * f) @ C.T
    D = 0.5 * (D + D.T)
    try:
        tr = np.trace(D @ S)
        if tr != 0 and np.isfinite(tr):
            D *= (float(nocc) / float(tr))
    except Exception:
        pass
    return D

print("Done reading the helper functions!")

# Build molecule
print("Building the CNT")
mol = gto.Mole(
    atom=atoms,
    basis=basis_set,
    unit='Angstrom',
    charge=charge,
    spin=spin
)
mol.build()
nmo, nelec, nocc = mol.nao_nr(), mol.nelectron, mol.nelectron // 2


# Helper function for proper vaccum sampling
def eval_nuclear_potential(mol_obj, pts_bohr, eps=1e-12):
    """
    Vectorized nuclear potential at pts (Bohr).
    Returns shape (npts,) with V_nuc = - sum_A Z_A / |r - R_A|.
    pts_bohr : array (npts,3) in Bohr
    """
    coords_bohr = mol_obj.atom_coords(unit='Bohr')    # (nat,3)
    charges = np.asarray([mol_obj.atom_charge(i) for i in range(mol_obj.natm)], dtype=float)  # ZA
    # Vectorized: r (npts,1,3) - RA (1,natom,3) -> rA (npts,natom,3)
    rA = pts_bohr[:, None, :] - coords_bohr[None, :, :]
    dist = np.linalg.norm(rA, axis=2)                 # (npts, nat)
    # avoid zero-distance
    dist = np.where(dist < eps, eps, dist)
    V_nuc = - np.sum(charges[None, :] / dist, axis=1) # (npts,)
    return V_nuc


# Track best converged density and MO energies
last_converged_dm = None
mo_e = np.array([])

print("Done building the CNT")

# Base RKS factory
def make_rks():
    # Using DF 
    mf = dft.RKS(mol, xc="wb97xv")
    mf = mf.density_fit()   # J only
    mf.with_df.k = None     # force exact K
    mf.with_df.auxbasis = "def2-universal-jfit"


    # Everything else stays identical:
    mf.xc = xc_functional
    mf.max_cycle = max_scf_cycles
    mf.conv_tol = conv_tol
    mf.max_memory = max_memory_mb
    mf.damp = 0.7
    mf.level_shift = 0.5
    mf.diis_space = 8
    if hasattr(mf, 'diis_start_cycle'):
        mf.diis_start_cycle = 20
    return mf


# LDA preconditioning
print("Running LDA preconditioning SCF (loose tolerance)...")
mf_lda = dft.RKS(mol)
mf_lda.xc = 'lda,vwn'
mf_lda.max_cycle = 200
mf_lda.conv_tol = 1e-3
mf_lda.damp = 0.7
mf_lda.level_shift = 0.5
mf_lda.diis = None   # disable DIIS properly
mf_lda.chkfile = os.path.join(results_dir, "lda_precond.chk")
lda_converged = mf_lda.kernel()

D_lda = None
if mf_lda.converged:
    print("  -> LDA converged, will use density as preconditioner")
    D_lda = mf_lda.make_rdm1()

# Initial guess chain
H_core, S = make_rks().get_hcore(), mol.intor("int1e_ovlp")
save_np_txt("H_core.txt", H_core); save_np_txt("S_overlap.txt", S)

print("Making the first guess...")

def try_sad():
    try:
        tmp = scf.RHF(mol)
        D_sad = tmp.get_init_guess(mol, "sad")
        if isinstance(D_sad, np.ndarray) and D_sad.shape == (nmo, nmo):
            return D_sad
        if isinstance(D_sad, np.ndarray) and D_sad.ndim == 2 and D_sad.shape[0] == nmo:
            C = D_sad
            if C.shape[1] >= nocc:
                return C[:, :nocc] @ C[:, :nocc].T
    except Exception as e:
        print("  -> SAD generation failed:", e)
    return None

D_sad = try_sad()
D_pur = None
if D_sad is not None:
    D_sad = safe_symmetrize_and_normalize(D_sad, S=S, nocc=nocc)
    D_pur, iters = mcweeny_purify(D_sad, S=S, nocc=nocc,
                                  max_iter=purify_max_iter, tol=purify_tol)
    if D_pur is not None:
        print("  -> Purification succeeded. iterations:", iters)
        save_np_txt("D_SAD_raw.txt", D_sad)
        save_np_txt("D_SAD_purified.txt", D_pur)

# Field loop
E0 = None
for i, (volt, E) in enumerate(zip(voltages, E_fields)):
    E_int = int(volt)                  # scalar integer used in filenames
    E_au = float(E) / AU_EFIELD_V_PER_M
    n = field_direction / np.linalg.norm(field_direction)

    print(f"\n== Field {E_int:.2e} V/m ==")
    chk = os.path.join(results_dir, f"scf_field_{int(volt)}.chk")
    mf = make_rks()
    mf.chkfile = chk

    # SANITY CHECK
    # ---------------------------------------------------
    print("=== Sanity check DF / JK setup ===")
    print("  with_df object:", mf.with_df)
    print("  is K disabled:", getattr(mf.with_df, "k", None))
    print("=== End sanity check ===")



    # Choose best available initial density
    dm0 = last_converged_dm if last_converged_dm is not None else (
        D_pur if D_pur is not None else (D_lda if D_lda is not None else None)
    )
    if dm0 is not None:
        print("  Using preconditioned / last-converged initial density for this field.")

    # Robust SCF restart: loading ONLY previous density (never MF object)
    if E > 0:
        prev = os.path.join(results_dir, f"scf_field_{int(voltages[i-1])}.chk")
        if os.path.exists(prev):
            try:
                # Load the density stored by PySCF into the checkpoint
                dm_prev = scf.chkfile.load(prev, 'scf/dm')

                # Convert tagged arrays or other containers to plain ndarray
                if dm_prev is not None: 
                    dm_prev = np.asarray(dm_prev)


                # Validate DM
                if (
                    isinstance(dm_prev, np.ndarray)
                    and dm_prev.shape == (nmo, nmo)
                    and np.all(np.isfinite(dm_prev))
                ):
                    dm_prev = safe_symmetrize_and_normalize(dm_prev, S=S, nocc=nocc)
                    dm0 = dm_prev
                    print("  Loaded density from checkpoint and will use it as dm0.")
                else:
                    print("  Checkpoint found but density invalid → starting with fresh guess.")

            except Exception as e:
                print("  Failed to restart from previous checkpoint:", e)

    # Two-stage SCF: loose tol then tight tol
    try:
        print("  Stage 1: loose SCF...")
        mf.conv_tol = 1e-3
        e1 = mf.kernel(dm0=dm0, efield=(0, 0, float(E_au)))
        print(f"    Stage 1 converged? {mf.converged}   "
              f"E = {e1 if e1 is not None else float('nan'):.12f} Ha")

        mo_e = np.asarray(mf.mo_energy) if getattr(mf, "mo_energy", None) is not None else np.array([])
        np.savetxt(os.path.join(results_dir, f"MO_energies_field{E_int}.txt"),
                   mo_e, fmt="%.12f")

        print("  Stage 2: tight SCF...")
        mf.conv_tol = 1e-6
        mf.diis_space = 8
        if hasattr(mf, 'diis_start_cycle'):
            mf.diis_start_cycle = 20
        e2 = mf.kernel(dm0=(mf.make_rdm1() if mf.converged else dm0),
                       efield=(0, 0, float(E_au)))
        print(f"    Stage 2 converged? {mf.converged}   "
              f"E = {e2 if e2 is not None else float('nan'):.12f} Ha")

        mo_e = np.asarray(mf.mo_energy) if getattr(mf, "mo_energy", None) is not None else np.array([])
        np.savetxt(os.path.join(results_dir, f"MO_energies_field{E_int}.txt"),
                   mo_e, fmt="%.12f")

        if mf.converged:
            last_converged_dm = mf.make_rdm1()
            dm_tagged = lib.tag_array(last_converged_dm, with_df=mf.with_df)
            scf.chkfile.dump(mf.chkfile, 'scf/dm', dm_tagged)


        if E == 0:
            E0 = e2 if e2 is not None else e1

        # ===============================================================
        # VACUUM SAMPLING MODULE (Hartree + XC + nuclear potential)
        # (Option 2: radial sampling normal to the tube axis)
        # CHUNKED VERSION FOR LARGE SYSTEMS
        # ===============================================================

        # Parameters
        r_start_ang = 25.0
        r_end_ang   = 60.0
        N_r = 18
        N_theta = 48
        take_fraction_far = 0.2
        CHUNK = 3000   # tweakable — safe for 45 CPUs and large molecules

        coords_A = np.array(coords_ang)
        z_min_ang = coords_A[:,2].min()
        z_max_ang = coords_A[:,2].max()
        z_mid_ang = 0.5 * (z_min_ang + z_max_ang)

        r_ang = np.linspace(r_start_ang, r_end_ang, N_r)
        theta = np.linspace(0.0, 2*np.pi, N_theta, endpoint=False)

        z_mid_au = z_mid_ang * ANG_TO_AU
        pts_list = []
        for r in r_ang:
            r_au = r * ANG_TO_AU
            xs = r_au * np.cos(theta)
            ys = r_au * np.sin(theta)
            zs = np.full_like(xs, z_mid_au)
            pts_list.append(np.column_stack([xs, ys, zs]))

        pts = np.vstack(pts_list)
        npts = pts.shape[0]

        dm_used = mf.make_rdm1()
        dm_used = np.asarray(dm_used, dtype=float)
        if not (isinstance(dm_used, np.ndarray) and dm_used.shape == (nmo, nmo) and np.all(np.isfinite(dm_used))):
            raise RuntimeError("dm_used invalid (non-finite or wrong shape) — aborting vacuum sampling")
        ni = mf._numint
        

        # Prepare result arrays (fill with NaN to spot uninitialized entries)
        rho_p     = np.empty(npts, dtype=float)
        rho_x_p   = np.empty(npts, dtype=float)
        rho_y_p   = np.empty(npts, dtype=float)
        rho_z_p   = np.empty(npts, dtype=float)
        tau_p     = np.empty(npts, dtype=float)
        vxc_grid  = np.empty(npts, dtype=float)
        v_hartree = np.empty(npts, dtype=float)
        v_nuc_p   = np.empty(npts, dtype=float)

        rho_p.fill(np.nan)
        rho_x_p.fill(np.nan)
        rho_y_p.fill(np.nan)
        rho_z_p.fill(np.nan)
        tau_p.fill(np.nan)
        vxc_grid.fill(np.nan)
        v_hartree.fill(np.nan)
        v_nuc_p.fill(np.nan)

        # --- Compute Coulomb J only ---
        print("  Computing DF-J (no K) for vacuum sampling...")
        vj_mat = mf.get_j(dm_used)
        vj_mat = np.asarray(vj_mat, float)

        nao_loc = mol.nao_nr()
        if vj_mat.shape != (nao_loc, nao_loc):
            raise RuntimeError(f"DF-J returned wrong shape: {vj_mat.shape}, expected {(nao_loc, nao_loc)})")

        # Diagnostics
        print("DEBUG vj_mat type:", type(vj_mat))
        print("DEBUG vj_mat shape:", vj_mat.shape)


        # ---- chunked loop ----
        for i0 in range(0, npts, CHUNK):
            i1 = min(i0 + CHUNK, npts)
            pts_chunk = pts[i0:i1]

            # AO and derivatives at points
            # Get AO values for deriv1 explicitly requesting 4 components
            nchunk = pts_chunk.shape[0]
            # ---- SAFE AO-derivative extraction (PySCF-proof) ----
            ao_all = mol.eval_gto("GTOval_sph_deriv1", pts_chunk, comp=4)
            ao_all = np.asarray(ao_all)

            # Flatten everything and rebuild robustly:
            ao_flat = ao_all.reshape(-1)

            # The correct size MUST be: ncomp * npts * nao_loc
            expected = 4 * nchunk * nao_loc
            if ao_flat.size != expected:
                raise RuntimeError(
                    f"AO derivative size mismatch: got {ao_flat.size}, "
                    f"expected {expected}. Raw shape: {ao_all.shape}"
                )

            # Reshape into (4, npts, nao)
            ao = ao_flat.reshape(4, nchunk, nao_loc)

            ao0 = ao[0]
            aox = ao[1]
            aoy = ao[2]
            aoz = ao[3]

            # Density (per-point)
            rho_c = np.einsum("pi,ij,pj->p", ao0, dm_used, ao0)

            # gradient of density
            rho_x_c = 2.0 * np.einsum("pi,ij,pj->p", ao0, dm_used, aox)
            rho_y_c = 2.0 * np.einsum("pi,ij,pj->p", ao0, dm_used, aoy)
            rho_z_c = 2.0 * np.einsum("pi,ij,pj->p", ao0, dm_used, aoz)

            # tau (kinetic energy density)
            tau_c = (
                np.einsum("pi,ij,pj->p", aox, dm_used, aox)
            + np.einsum("pi,ij,pj->p", aoy, dm_used, aoy)
            + np.einsum("pi,ij,pj->p", aoz, dm_used, aoz)
            )

            # Save MGGA ingredients
            rho_p[i0:i1]   = rho_c
            rho_x_p[i0:i1] = rho_x_c
            rho_y_p[i0:i1] = rho_y_c
            rho_z_p[i0:i1] = rho_z_c
            tau_p[i0:i1]   = tau_c

            # XC (chunked)
            # wb97x is GGA → NO tau
            rho_gga_c = np.vstack([rho_c, rho_x_c, rho_y_c, rho_z_c])
            rho_for_xc = rho_gga_c[None, :, :]   # shape (1, 4, n_chunk)


            # Strip the nonlocal VV10 term when evaluating XC pointwise
            xc_local = "wb97x"
            exc_val, vxc_val, *rest = ni.eval_xc(xc_local, rho_for_xc, deriv=1)


            # ---- robust extraction of scalar v_xc (pointwise) ----
            # vxc_val may be:
            # - a list/tuple where first element is ndarray (ncomp, npoints) or (npoints,)
            # - a ndarray with shape (ncomp, npoints) or (npoints,)
            # We want the scalar channel: the derivative w.r.t. rho, which is channel 0.
            if isinstance(vxc_val, (list, tuple)):
                # first entry often contains the derivative array
                v_arr = np.asarray(vxc_val[0])
            else:
                v_arr = np.asarray(vxc_val)

            # v_arr can be (npoints,) or (ncomp, npoints) or (1, npoints)
            if v_arr.ndim == 1:
                # already scalar per-point
                v_block = v_arr.copy()
            elif v_arr.ndim == 2:
                # take first channel (dv/drho) which is v_arr[0]
                v_block = v_arr[0].ravel()
            else:
                # unexpected shape: try to flatten conservatively then slice
                v_block = v_arr.ravel()
                v_block = v_block[: (i1 - i0)]

            # ensure correct length for this chunk
            if v_block.size != (i1 - i0):
                if v_block.size > (i1 - i0):
                    v_block = v_block[: (i1 - i0)]
                else:
                    # pad with last finite value or nan
                    if v_block.size > 0 and np.any(np.isfinite(v_block)):
                        last = np.nanmean(v_block[np.isfinite(v_block)])
                        pad = np.full((i1 - i0 - v_block.size,), last, dtype=float)
                    else:
                        pad = np.full((i1 - i0 - v_block.size,), 0.0, dtype=float)
                    v_block = np.concatenate([v_block, pad])

            # assign the pointwise scalar v_xc for this chunk
            vxc_grid[i0:i1] = v_block

            # If you also want the XC energy density per point available (not integrated here),
            # get the exc (energy density) scalar channel similarly:
            # exc_val may be shape (1, npoints) or (npoints,)
            if isinstance(exc_val, (list, tuple)):
                exc_arr = np.asarray(exc_val[0])
            else:
                exc_arr = np.asarray(exc_val)
            if exc_arr.ndim == 2:
                exc_per_point = exc_arr[0].ravel()
            elif exc_arr.ndim == 1:
                exc_per_point = exc_arr
            else:
                exc_per_point = exc_arr.ravel()[: (i1 - i0)]
            # store if you need it later (optional)
            # exc_chunk[i0:i1] = exc_per_point   # uncomment if you created exc_chunk earlier

            # Hartree potential for chunk (using precomputed vj_mat)
            try:
                v_hartree[i0:i1] = np.einsum("pi,ij,pj->p", ao0, vj_mat, ao0)
            except Exception:
                v_hartree[i0:i1] = 0.0

            # Nuclear potential for chunk (robust helper)
            v_nuc_p[i0:i1] = eval_nuclear_potential(mol, pts_chunk)
            
            print("DEBUG AO shapes:", ao_all.shape)  # should be (4, n_chunk, nao)



        # ---- assemble final V_loc (scalar) ----
        # replace possible NaNs in components by local finite-means to avoid propagating NaN
        finite_mask = np.isfinite(v_hartree) & np.isfinite(vxc_grid) & np.isfinite(v_nuc_p)
        if not np.any(finite_mask):
            raise RuntimeError("All vacuum sampling potentials are NaN — check interpolator / eval functions.")
        # replace NaNs individually with local finite mean
        if not np.all(np.isfinite(v_hartree)):
            v_hartree[~np.isfinite(v_hartree)] = np.nanmean(v_hartree[np.isfinite(v_hartree)])
        if not np.all(np.isfinite(vxc_grid)):
            vxc_grid[~np.isfinite(vxc_grid)] = np.nanmean(vxc_grid[np.isfinite(vxc_grid)])
        if not np.all(np.isfinite(v_nuc_p)):
            v_nuc_p[~np.isfinite(v_nuc_p)] = np.nanmean(v_nuc_p[np.isfinite(v_nuc_p)])

        v_loc = v_hartree + vxc_grid + v_nuc_p

         # ---- Fix 2: subtract asymptotic Coulomb offset (finite-system gauge) ----
        n_far_r = max(1, int(np.ceil(take_fraction_far * N_r)))
        far_start = (N_r - n_far_r) * N_theta
        far_mask = np.zeros(v_hartree.size, dtype=bool)
        far_mask[far_start:] = True

        # compute average far-field electrostatic potential (Hartree + nuclear)
        coul_far = np.mean((v_hartree + v_nuc_p)[far_mask])

        # subtract this constant offset from the *total* local potential
        v_loc = v_loc - coul_far

        # ---- radial averaging ----
        if v_loc.size != (N_r * N_theta):
            raise RuntimeError(f"v_loc length {v_loc.size} incompatible with N_r*N_theta = {N_r * N_theta}")

        v_loc_by_r = v_loc.reshape((N_r, N_theta)).mean(axis=1)

        n_tail = max(1, int(np.ceil(take_fraction_far * N_r)))
        vacuum_level = float(np.mean(v_loc_by_r[-n_tail:]))

        radii_A = np.repeat(r_ang, N_theta)
        thetas  = np.tile(theta, N_r)

        np.savetxt(
            os.path.join(results_dir, f"vacuum_profile_field{E_int}.txt"),
            np.column_stack([radii_A, thetas, pts[:,0], pts[:,1], pts[:,2], v_loc]),
            header="r_ang theta rad_x(bohr) rad_y(bohr) z(bohr) V_loc(Ha)"
        )

        print(f"  Vacuum level extracted (Ha): {vacuum_level:.12f}")



    except Exception as e:
        print("  SCF raised exception:", e)
        continue

    # === outputs ===
    Hc = mf.get_hcore()
    So = mol.intor("int1e_ovlp")
    Df = mf.make_rdm1()

    save_np_txt(f"H_core_field{int(E_int)}.txt", Hc)
    save_np_txt(f"S_overlap_field{int(E_int)}.txt", So)
    save_np_txt(f"D_density_field{int(E_int)}.txt", Df)

    mo_e = np.asarray(mf.mo_energy) if getattr(mf, "mo_energy", None) is not None else np.array([])
    homo = mo_e[nocc-1] if len(mo_e) >= nocc else np.nan
    lumo = mo_e[nocc] if len(mo_e) > nocc else np.nan
    gap = lumo - homo
    Ha_to_eV = 27.211386

    # Now we add the ionization potential (IP = V_vac - E_HOMO); all in Ha
    try:
        IP = vacuum_level - homo
        with open(os.path.join(results_dir,
                f"IP_field{int(E_int)}.txt"), "w") as f_ip:
            f_ip.write(f"IP_Ha {IP:.12f}\nIP_eV {IP * Ha_to_eV:.6f}\n")
        print(f"  Ionization potential saved (Ha): {IP:.12f}  (eV: {IP * Ha_to_eV:.6f})")
    except Exception as e:
        print("  Ionization potential calculation failed:", e)

    with open(os.path.join(results_dir, f"HOMO_LUMO_field{E_int}.txt"), "w") as f:
        f.write(f"HOMO_Ha {homo:.12f}\nHOMO_eV {homo * Ha_to_eV:.6f}\n")
        f.write(f"LUMO_Ha {lumo:.12f}\nLUMO_eV {lumo * Ha_to_eV:.6f}\n")
        f.write(f"GAP_Ha {gap:.12f}\nGAP_eV {gap * Ha_to_eV:.6f}\n")

    with open(os.path.join(results_dir, f"MO_info_field{int(E_int)}.txt"), "w") as f:
        f.write(f"E_tot {getattr(mf, 'e_tot', float('nan')):.12f} Ha\n")
        f.write(f"HOMO {homo:.12f} Ha\nLUMO {lumo:.12f} Ha\n")
        f.write(f"GAP {gap:.12f} Ha\n")

#  Simulate the properties under the interference of the electric field
stabilization_energies = []
work_functions = []

Ha_to_eV = 27.211386  # ensure defined in this scope

for j, E in enumerate(E_fields[1:]):  # skip the 0 field for stabilization
    print(f"\n[Scanner-like loop] Applying field {E:.2e} V/m")
    # pick corresponding voltage integer (same ordering as voltages)
    volt = voltages[j+1]
    E_int = int(volt)
    chk_field = os.path.join(results_dir, f"scf_field_{int(volt)}.chk")
    mf_field = make_rks()
    mf_field.chkfile = chk_field

    # Reuse best available initial density
    dm0 = last_converged_dm if last_converged_dm is not None else (
        D_pur if D_pur is not None else (D_lda if D_lda is not None else None)
    )
    if dm0 is not None:
        print("  Using preconditioned / last-converged initial density for this field.")

    try:
        print("  Stage 1: loose SCF for stabilization...")
        mf_field.conv_tol = 1e-3
        e1 = mf_field.kernel(dm0=dm0, efield=(0, 0, E * V_PER_M_TO_AU))
        print(f"    Stage 1 converged? {mf_field.converged}   "
              f"E = {e1 if e1 is not None else float('nan'):.12f} Ha")

        print("  Stage 2: tight SCF for stabilization...")
        mf_field.conv_tol = 1e-6
        mf_field.diis_space = 8
        if hasattr(mf_field, 'diis_start_cycle'):
            mf_field.diis_start_cycle = 20
        e2 = mf_field.kernel(dm0=(mf_field.make_rdm1() if mf_field.converged else dm0),
                             efield=(0, 0, E * V_PER_M_TO_AU))
        print(f"    Stage 2 converged? {mf_field.converged}   "
              f"E = {e2 if e2 is not None else float('nan'):.12f} Ha")

        # Pick best available SCF energy
        Etot_field = e2 if e2 is not None else e1
        if Etot_field is None:
            print("  Warning: No SCF energy obtained for this field, skipping.")
            stabilization_energies.append(np.nan)
            work_functions.append(np.nan)
            continue

        # Update last converged density for next reuse
        if mf_field.converged:
            last_converged_dm = mf_field.make_rdm1()

        # HOMO energy
        if getattr(mf_field, "mo_energy", None) is not None and len(mf_field.mo_energy) > nocc - 1:
            homo_E = mf_field.mo_energy[nocc - 1]
        else:
            homo_E = np.nan

        # For the stabilization loop we use the vacuum_level previously computed for each field.
        # We saved vacuum per field earlier (vacuum_profile_field{E_int}.txt). Re-load it to be robust:
        vacuum_level = np.nan  # fallback
        vac_file = os.path.join(results_dir, f"vacuum_profile_field{E_int}.txt")
        if os.path.exists(vac_file):
            try:
                # file columns expected: r_ang theta rad_x rad_y z V_loc
                data = np.loadtxt(vac_file)
                if data.ndim == 1:
                    # single-line file: ensure shape (1, ncols)
                    data = data.reshape(1, -1)
                # compute vacuum by averaging V_loc at the largest radii rows
                radii = data[:, 0]
                Vloc = data[:, -1]
                # pick rows where radius is within top 20%
                thr = np.percentile(radii, 80)
                sel = radii >= thr
                if np.any(sel):
                    vacuum_level = float(np.mean(Vloc[sel]))
                else:
                    vacuum_level = float(np.mean(Vloc))
            except Exception as exc:
                print(f"  Warning: could not read/parse {vac_file}: {exc}")
                vacuum_level = np.nan
        else:
            print(f"  Warning: vacuum file {vac_file} not found. vacuum_level undefined for this field.")

        # If either vacuum_level or homo_E is not finite, skip producing a work function
        if not np.isfinite(vacuum_level) or not np.isfinite(homo_E):
            print("  Warning: vacuum_level or HOMO undefined/NaN — skipping work-function for this field.")
            wf_Ha = np.nan
            wf_eV = np.nan
        else:
            wf_Ha = vacuum_level - homo_E
            wf_eV = wf_Ha * Ha_to_eV

        # Save work function file (Ha and eV) even if NaN — keeps outputs consistent per-field
        try:
            np.savetxt(os.path.join(results_dir, f"work_function_field{E_int}.txt"),
                       np.array([wf_Ha, wf_eV]), header="WF_Ha WF_eV")
        except Exception as exc:
            print(f"  Warning: failed to save work function file for field {E_int}: {exc}")

        # ensure E0 is set for stabilization energy
        if E0 is None:
            print("  Warning: E0 (zero-field reference energy) is not available — skipping stabilization/work function for this field.")
            stabilization_energies.append(np.nan)
            work_functions.append(np.nan)
            continue

        # stabilization energy (Ha)
        delta_E = Etot_field - E0
        stabilization_energies.append(delta_E)
        work_functions.append(wf_Ha)

        # convert last appended values to eV (in-place)
        stabilization_energies[-1] *= Ha_to_eV
        if np.isfinite(work_functions[-1]):
            work_functions[-1] *= Ha_to_eV
        else:
            work_functions[-1] = np.nan

    except Exception as e:
        print(f"  Stabilization SCF failed at E={E:.2e}: {e}")
        stabilization_energies.append(np.nan)
        work_functions.append(np.nan)
        continue


# Proper scientific notation format 
def sci_fmt(val, pos):
    if val == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / 10**exponent
    return r"${:.2f}\times 10^{{{:d}}}$".format(mantissa, exponent)

# Saving the numerical data
np.savetxt("stabilization_energy_CNT(7,5).txt", np.column_stack([E_fields[1:], stabilization_energies]),
           header="Field (V/m)    Stabilization Energy (eV)")

np.savetxt("work_function_Efield_CNT(7,5).txt", np.column_stack([E_fields[1:], work_functions]),
           header="Field (V/m)    Work Function (eV)")

# Save last MO energies
try:
    np.savetxt(os.path.join(results_dir, f"MO_energies_field{int(voltages[-1])}.txt"), mo_e, fmt="%.12f")
except Exception:
    pass


# Computing Mulliken atomic charges
print("\n[3.1] Computing Mulliken atomic charges...")
pop, charges = mf.mulliken_pop()

# Save Mulliken charges to file
with open("mulliken_charges_CNT_7,5.txt", "w") as f:
    f.write("Atom_Index\tAtom_Label\tCharge\n")
    for i, charge in enumerate(charges):
        atom = mol.atom_symbol(i)
        f.write(f"{i+1}\t{atom}\t{charge:.6f}\n")

# Bar plot of Mulliken charges
print("\n[3.2] Plotting Mulliken atomic charge distribution...")
atoms = [mol.atom_symbol(i) + str(i+1) for i in range(mol.natm)]

plt.figure(figsize=(16, 6))
plt.bar(atoms, charges, color='steelblue')
plt.xticks(rotation=90)
plt.xlabel("Atoms")
plt.ylabel("Mulliken Charge (e)")
plt.title("Mulliken Atomic Charges - CNT (5,5)")
plt.tight_layout()
plt.savefig("mulliken_charges_barplot_7,5.png", dpi=300)
plt.close()
print("  -> Mulliken charge barplot saved as MCharges_75.png")

# Plotting individual property graphs
plt.figure()
plt.plot(E_fields[1:]*1e-6, stabilization_energies, 'o-', color='blue')
plt.xlabel("Electric Field (V/µm)")
plt.ylabel("Stabilization Energy (eV)")
plt.title("Stabilization Energy vs Electric Field")
ax = plt.gca()
ax.tick_params(axis='both', labelsize=9)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt))
plt.grid(False)
plt.savefig("Stabilization_vs_Field_CNT(7,5).png", dpi=500)
plt.close()

plt.figure()
plt.plot(E_fields[1:]*1e-6, work_functions, 'd-.', color='red')
plt.xlabel("Electric Field (V/µm)")
plt.ylabel("Work Function (eV)")
plt.title("Work Function vs Electric Field")
ax = plt.gca()
ax.tick_params(axis='both', labelsize=9)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt))
plt.grid(False)
plt.savefig("WorkFunction_vs_Field_CNT(7,5).png", dpi=500)
plt.close()


print("Done, dude!")

