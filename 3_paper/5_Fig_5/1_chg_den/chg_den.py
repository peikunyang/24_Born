#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import MDAnalysis as mda


SYSTEMS = {
    "1_K": {
        "title": "K",
        "pdb": "single_K_sel21A_1118w.pdb",
        "ion_resnames": {"POT"},
        "ion_names": {"K", "POT"},
        "charges": [
            ("q0.0", "0"),
            ("q0.5", "0.5"),
            ("q1.0", "1"),
        ],
    },
    "2_Cl": {
        "title": "CL",
        "pdb": "single_Cl_sel21A_1118w.pdb",
        "ion_resnames": {"CLA"},
        "ion_names": {"CL", "CLA"},
        "charges": [
            ("q0.0", "0"),
            ("qm0.5", "-0.5"),
            ("qm1.0", "-1"),
        ],
    },
    "3_Mg": {
        "title": "MG",
        "pdb": "single_Mg_sel21A_1118w.pdb",
        "ion_resnames": {"MG"},
        "ion_names": {"MG"},
        "charges": [
            ("q0.0", "0"),
            ("q1.0", "1"),
            ("q2.0", "2"),
        ],
    },
}

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL"}
O_NAMES = {"O", "OH2", "OW", "OT"}


def find_ion_index(u, ion_resnames, ion_names):
    candidates = []
    for atom in u.atoms:
        if atom.resname in ion_resnames or atom.name.upper() in ion_names:
            candidates.append(atom.index)

    if len(candidates) != 1:
        raise RuntimeError(f"ion 數量錯誤: {candidates}")

    return candidates[0]


def find_water_atoms(u, ion_index):
    indices = []
    charges = []

    for atom in u.atoms:
        if atom.index == ion_index:
            continue

        if atom.resname.upper() not in WATER_RESNAMES:
            continue

        name = atom.name.upper()

        if name in O_NAMES:
            indices.append(atom.index)
            charges.append(-0.834)
        elif name.startswith("H"):
            indices.append(atom.index)
            charges.append(0.417)

    if len(indices) == 0:
        raise RuntimeError("找不到 water atoms")

    return np.array(indices, dtype=int), np.array(charges, dtype=float)


def calc_shell_charge_density(
    pdb_path,
    dcd_path,
    ion_resnames,
    ion_names,
    r_min,
    r_max,
    dr,
    stride,
):
    u = mda.Universe(str(pdb_path), str(dcd_path))

    ion_index = find_ion_index(u, ion_resnames, ion_names)
    water_indices, water_charges = find_water_atoms(u, ion_index)

    r_values = np.arange(r_min, r_max + 0.5 * dr, dr)
    edges = np.arange(r_min - 0.5 * dr, r_max + dr, dr)

    charge_sum = np.zeros(len(r_values), dtype=float)
    n_frames = 0

    for ts in u.trajectory[::stride]:
        ion_pos = u.atoms[ion_index].position
        water_pos = u.atoms[water_indices].positions

        dist = np.linalg.norm(water_pos - ion_pos, axis=1)

        hist, _ = np.histogram(
            dist,
            bins=edges,
            weights=water_charges,
        )

        charge_sum += hist
        n_frames += 1

    if n_frames == 0:
        raise RuntimeError(f"沒有讀到 frame: {dcd_path}")

    # 這裡算的是 4*pi*r^2*rho(q,r)，單位 e/A
    density = charge_sum / n_frames / dr

    return r_values, density, n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md_root", required=True)
    parser.add_argument("--cluster_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--r_min", type=float, default=1.0)
    parser.add_argument("--r_max", type=float, default=15.0)
    parser.add_argument("--dr", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    md_root = Path(args.md_root)
    cluster_dir = Path(args.cluster_dir)

    blocks = []
    r_ref = None

    for sys_name, info in SYSTEMS.items():
        pdb_path = cluster_dir / info["pdb"]
        sys_md_dir = md_root / sys_name / "MD"

        curves = []

        print(f"\n=== {sys_name} ===")

        for q_dir, q_label in info["charges"]:
            dcd_path = sys_md_dir / q_dir / "traj.dcd"

            if not pdb_path.is_file():
                raise FileNotFoundError(pdb_path)

            if not dcd_path.is_file():
                raise FileNotFoundError(dcd_path)

            print(f"reading {dcd_path}")

            r_values, density, n_frames = calc_shell_charge_density(
                pdb_path=pdb_path,
                dcd_path=dcd_path,
                ion_resnames=info["ion_resnames"],
                ion_names=info["ion_names"],
                r_min=args.r_min,
                r_max=args.r_max,
                dr=args.dr,
                stride=args.stride,
            )

            if r_ref is None:
                r_ref = r_values
            else:
                if len(r_ref) != len(r_values) or np.max(np.abs(r_ref - r_values)) > 1e-8:
                    raise RuntimeError("r grid 不一致")

            curves.append((q_label, density))

            print(f"done {q_dir}, frames = {n_frames}")

        blocks.append((info["title"], curves))

    rows = []

    header = []
    for i, (title, curves) in enumerate(blocks):
        if i > 0:
            header.append("")
        header.extend([title, "Div"])
        header.extend([q_label for q_label, _ in curves])
    rows.append(header)

    for idx, r in enumerate(r_ref):
        row = []

        for i, (title, curves) in enumerate(blocks):
            if i > 0:
                row.append("")

            row.append("")
            row.append(f"{r:g}")

            for q_label, density in curves:
                row.append(f"{density[idx]:.3f}")

        rows.append(row)

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nwritten: {args.out_csv}")


if __name__ == "__main__":
    main()

