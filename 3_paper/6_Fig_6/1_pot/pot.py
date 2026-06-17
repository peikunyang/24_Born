#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import MDAnalysis as mda


COULOMB = 332.06371

SYSTEMS = {
    "4_K_K": {
        "pdb_prefix": "pair_K_K",
        "ion_resnames": ["POT", "POT"],
        "ion_labels": ["K1", "K2"],
        "out_name": "4_K_K_cumulative_potential.csv",
    },
    "4_K_K_Q0": {
        "pdb_prefix": "pair_K_K",
        "ion_resnames": ["POT", "POT"],
        "ion_labels": ["K1", "K2"],
        "out_name": "4_K_K_Q0_cumulative_potential.csv",
    },
    "5_CL_CL": {
        "pdb_prefix": "pair_Cl_Cl",
        "ion_resnames": ["CLA", "CLA"],
        "ion_labels": ["CL1", "CL2"],
        "out_name": "5_CL_CL_cumulative_potential.csv",
    },
    "5_CL_CL_Q0": {
        "pdb_prefix": "pair_Cl_Cl",
        "ion_resnames": ["CLA", "CLA"],
        "ion_labels": ["CL1", "CL2"],
        "out_name": "5_CL_CL_Q0_cumulative_potential.csv",
    },
    "6_K_CL": {
        "pdb_prefix": "pair_K_Cl",
        "ion_resnames": ["POT", "CLA"],
        "ion_labels": ["K", "CL"],
        "out_name": "6_K_CL_cumulative_potential.csv",
    },
    "6_K_CL_Q0": {
        "pdb_prefix": "pair_K_Cl",
        "ion_resnames": ["POT", "CLA"],
        "ion_labels": ["K", "CL"],
        "out_name": "6_K_CL_Q0_cumulative_potential.csv",
    },
}

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL"}
O_NAMES = {"O", "OH2", "OW", "OT"}


def find_two_ions(u, ion_resnames):
    used = set()
    indices = []

    for resname in ion_resnames:
        found = None

        for atom in u.atoms:
            if atom.index in used:
                continue

            if atom.resname.upper() == resname.upper():
                found = atom.index
                break

        if found is None:
            raise RuntimeError(f"Cannot find ion with resname={resname}")

        indices.append(found)
        used.add(found)

    return indices[0], indices[1]


def find_water_atoms(u):
    indices = []
    charges = []

    for atom in u.atoms:
        if atom.resname.upper() not in WATER_RESNAMES:
            continue

        name = atom.name.upper()

        if name in O_NAMES:
            indices.append(atom.index)
            charges.append(-0.834)
        elif name.startswith("H"):
            indices.append(atom.index)
            charges.append(0.417)

    if not indices:
        raise RuntimeError("Cannot find water atoms")

    return np.array(indices, dtype=int), np.array(charges, dtype=float)


def calc_one_d(pdb_path, dcd_path, ion_resnames, r_min, r_max, dr, stride):
    u = mda.Universe(str(pdb_path), str(dcd_path))

    ion1_idx, ion2_idx = find_two_ions(u, ion_resnames)
    water_indices, water_charges = find_water_atoms(u)

    # r is the upper radius of the cumulative region.
    # The radius is measured from the ion pair center.
    n_bins = int(round(r_max / dr))

    if abs(n_bins * dr - r_max) > 1e-8:
        raise ValueError("r_max must be an integer multiple of dr")

    # Histogram bins are 0 to r_max.
    # A tiny extension of the last edge includes atoms exactly at r_max.
    edges = np.linspace(0.0, r_max, n_bins + 1)
    edges_for_hist = edges.copy()
    edges_for_hist[-1] += 1e-8

    # r values are 0.1, 0.2, ..., r_max when dr = 0.1.
    all_r_values = np.arange(1, n_bins + 1, dtype=float) * dr

    keep = all_r_values >= r_min - 1e-12
    r_values = all_r_values[keep]

    phi1_sum = np.zeros(len(r_values), dtype=float)
    phi2_sum = np.zeros(len(r_values), dtype=float)

    n_frames = 0

    for ts in u.trajectory[::stride]:
        ion1_pos = u.atoms[ion1_idx].position.astype(float)
        ion2_pos = u.atoms[ion2_idx].position.astype(float)

        origin = 0.5 * (ion1_pos + ion2_pos)

        water_pos = u.atoms[water_indices].positions.astype(float)

        r_origin = np.linalg.norm(water_pos - origin, axis=1)
        r_ion1 = np.linalg.norm(water_pos - ion1_pos, axis=1)
        r_ion2 = np.linalg.norm(water_pos - ion2_pos, axis=1)

        # Atomic charge summation.
        # Units are kcal/(mol•e), because charge is in e and distance is in Å.
        w1 = COULOMB * water_charges / r_ion1
        w2 = COULOMB * water_charges / r_ion2

        # Shell contribution with respect to the ion pair center.
        hist1, _ = np.histogram(r_origin, bins=edges_for_hist, weights=w1)
        hist2, _ = np.histogram(r_origin, bins=edges_for_hist, weights=w2)

        if len(hist1) != n_bins or len(hist2) != n_bins:
            raise RuntimeError("Unexpected histogram size")

        # Convert shell potential to cumulative potential.
        cum1 = np.cumsum(hist1)
        cum2 = np.cumsum(hist2)

        phi1_sum += cum1[keep]
        phi2_sum += cum2[keep]

        n_frames += 1

    if n_frames == 0:
        raise RuntimeError(f"No frames read: {dcd_path}")

    return r_values, phi1_sum / n_frames, phi2_sum / n_frames


def write_system_csv(system_name, info, md_root, cluster_dir, out_dir, r_min, r_max, dr, stride):
    d_labels = [f"d{i:02d}" for i in range(1, 9)]

    all_curves = []
    r_ref = None

    for d_label in d_labels:
        d_num = int(d_label[1:])

        pdb_path = cluster_dir / f"{info['pdb_prefix']}_d{d_num:02d}A_sel21A_1117w.pdb"
        dcd_path = md_root / system_name / "MD" / d_label / "traj.dcd"

        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)

        if not dcd_path.is_file():
            raise FileNotFoundError(dcd_path)

        print(f"reading {system_name} {d_label}")

        r_values, phi1, phi2 = calc_one_d(
            pdb_path=pdb_path,
            dcd_path=dcd_path,
            ion_resnames=info["ion_resnames"],
            r_min=r_min,
            r_max=r_max,
            dr=dr,
            stride=stride,
        )

        if r_ref is None:
            r_ref = r_values
        else:
            if len(r_ref) != len(r_values) or np.max(np.abs(r_ref - r_values)) > 1e-8:
                raise RuntimeError("r grid is inconsistent")

        all_curves.append((d_label, phi1, phi2))

    rows = []

    header = ["r"]
    for d_label, phi1, phi2 in all_curves:
        header.append(f"{d_label}_{info['ion_labels'][0]}")
        header.append(f"{d_label}_{info['ion_labels'][1]}")
    rows.append(header)

    for i, r in enumerate(r_ref):
        row = [f"{r:g}"]

        for d_label, phi1, phi2 in all_curves:
            row.append(f"{phi1[i]:.6f}")
            row.append(f"{phi2[i]:.6f}")

        rows.append(row)

    out_path = out_dir / info["out_name"]

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"written: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cumulative water potential for pair ion trajectories"
    )

    parser.add_argument(
        "--md_root",
        required=True,
        help="Directory containing 4_K_K, 4_K_K_Q0, 5_CL_CL, 5_CL_CL_Q0, 6_K_CL, and 6_K_CL_Q0",
    )
    parser.add_argument(
        "--cluster_dir",
        required=True,
        help="Directory containing pair PDB files",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(SYSTEMS.keys()),
        choices=list(SYSTEMS.keys()),
        help="Systems to analyze",
    )
    parser.add_argument("--r_min", type=float, default=0.1)
    parser.add_argument("--r_max", type=float, default=15.0)
    parser.add_argument("--dr", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    md_root = Path(args.md_root)
    cluster_dir = Path(args.cluster_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for system_name in args.systems:
        write_system_csv(
            system_name=system_name,
            info=SYSTEMS[system_name],
            md_root=md_root,
            cluster_dir=cluster_dir,
            out_dir=out_dir,
            r_min=args.r_min,
            r_max=args.r_max,
            dr=args.dr,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()

