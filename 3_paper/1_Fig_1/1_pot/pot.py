#!/usr/bin/env python3
"""
Calculate basic water-to-ion electrostatic potential from DCD trajectories.

This script does NOT:
  - subtract q0.0
  - calculate Delta Phi
  - plot figures

It only calculates:
  Phi_water_to_ion(q, r) = 332.06371 * sum_{water atoms with distance <= r} q_water / distance

Unit:
  kcal/mol/e

Expected input structure:
  md_root/
    1_K/MD/q0.0/traj.dcd
    1_K/MD/q0.1/traj.dcd
    ...
    2_Cl/MD/q0.0/traj.dcd
    2_Cl/MD/qm0.1/traj.dcd
    ...
    3_Mg/MD/q0.0/traj.dcd
    3_Mg/MD/q0.2/traj.dcd
    ...

  cluster_dir/
    single_K_sel21A_1118w.pdb
    single_Cl_sel21A_1118w.pdb
    single_Mg_sel21A_1118w.pdb
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit(
        "ERROR: MDAnalysis is not installed.\n"
        "Install it first, for example:\n"
        "  conda install -c conda-forge mdanalysis\n"
        "or\n"
        "  pip install MDAnalysis\n"
    )


COULOMB_KCAL_MOL_A_PER_E2 = 332.06371

SYSTEMS = {
    "1_K": {
        "ion_label": "K",
        "pdb": "single_K_sel21A_1118w.pdb",
        "ion_resnames": {"POT"},
        "ion_names": {"K", "POT"},
    },
    "2_Cl": {
        "ion_label": "Cl",
        "pdb": "single_Cl_sel21A_1118w.pdb",
        "ion_resnames": {"CLA"},
        "ion_names": {"CL", "CLA"},
    },
    "3_Mg": {
        "ion_label": "Mg",
        "pdb": "single_Mg_sel21A_1118w.pdb",
        "ion_resnames": {"MG"},
        "ion_names": {"MG"},
    },
}

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL"}
OXYGEN_NAMES = {"O", "OH2", "OW", "OT"}
HYDROGEN_NAMES = {"H", "H1", "H2", "H3", "HW1", "HW2"}

TIP3P_CHARGES = {
    "O": -0.834,
    "OH2": -0.834,
    "OW": -0.834,
    "OT": -0.834,
    "H": 0.417,
    "H1": 0.417,
    "H2": 0.417,
    "H3": 0.417,
    "HW1": 0.417,
    "HW2": 0.417,
}


def parse_q_label(label: str) -> float:
    """
    Convert folder name to charge value.

    q0.1  ->  0.1
    q1.0  ->  1.0
    qm0.1 -> -0.1
    """
    if label.startswith("qm"):
        return -float(label[2:])
    if label.startswith("q"):
        return float(label[1:])
    raise ValueError(f"Cannot parse charge folder label: {label}")


def q_sort_key(path: Path) -> float:
    return parse_q_label(path.name)


def find_charge_dirs(system_md_dir: Path) -> List[Path]:
    charge_dirs = []
    for p in system_md_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("q"):
            continue
        if (p / "traj.dcd").is_file():
            charge_dirs.append(p)

    if not charge_dirs:
        raise FileNotFoundError(f"No q*/traj.dcd found in {system_md_dir}")

    return sorted(charge_dirs, key=q_sort_key)


def find_ion_index(u: mda.Universe, ion_resnames: set, ion_names: set) -> int:
    candidates = []
    for atom in u.atoms:
        if atom.resname in ion_resnames or atom.name.upper() in ion_names:
            candidates.append(atom.index)

    if len(candidates) == 1:
        return candidates[0]

    # Fallback: your PDB files put the ion as the first atom.
    first = u.atoms[0]
    if first.resname in ion_resnames or first.name.upper() in ion_names:
        return int(first.index)

    raise RuntimeError(
        f"Expected exactly one ion atom, found {len(candidates)} candidates: {candidates}"
    )


def find_water_atoms_and_charges(
    u: mda.Universe,
    ion_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      water_indices: all water O/H atom indices
      charges: corresponding TIP3P charges
      water_oxygen_indices: water oxygen atom indices, used only for coordinate sanity check
    """
    indices = []
    charges = []
    oxygen_indices = []

    for atom in u.atoms:
        if atom.index == ion_index:
            continue

        atom_name = atom.name.upper()
        resname = atom.resname.upper()

        if resname not in WATER_RESNAMES:
            continue

        if atom_name in OXYGEN_NAMES:
            indices.append(atom.index)
            charges.append(-0.834)
            oxygen_indices.append(atom.index)
        elif atom_name in HYDROGEN_NAMES or atom_name.startswith("H"):
            indices.append(atom.index)
            charges.append(0.417)

    if not indices:
        raise RuntimeError("No water atoms found.")

    return (
        np.asarray(indices, dtype=int),
        np.asarray(charges, dtype=float),
        np.asarray(oxygen_indices, dtype=int),
    )


def detect_coordinate_scale(
    u: mda.Universe,
    ion_index: int,
    water_oxygen_indices: np.ndarray,
    auto_scale: bool,
    user_scale: float | None,
) -> Tuple[float, float]:
    """
    DCD coordinates may be read as Angstrom or nm depending on software conventions.
    For this water cluster, max O-ion distance should be about 20 A.

    If max distance is about 2, assume nm and multiply by 10.
    """
    if user_scale is not None:
        scale = float(user_scale)
    else:
        scale = 1.0

    u.trajectory[0]
    ion_pos = u.atoms[ion_index].position.astype(float)
    o_pos = u.atoms[water_oxygen_indices].positions.astype(float)
    distances = np.linalg.norm(o_pos - ion_pos, axis=1)
    max_o_distance_raw = float(np.max(distances))

    if user_scale is None and auto_scale:
        if max_o_distance_raw < 5.0:
            scale = 10.0
        else:
            scale = 1.0

    return scale, max_o_distance_raw


def calculate_potential_for_dcd(
    pdb_path: Path,
    dcd_path: Path,
    system_info: Dict,
    r_max: float,
    dr: float,
    stride: int,
    start: int | None,
    stop: int | None,
    auto_coord_scale: bool,
    coord_scale: float | None,
    progress_every: int,
) -> Dict:
    u = mda.Universe(str(pdb_path), str(dcd_path))

    ion_index = find_ion_index(
        u,
        ion_resnames=system_info["ion_resnames"],
        ion_names=system_info["ion_names"],
    )

    water_indices, water_charges, water_oxygen_indices = find_water_atoms_and_charges(
        u, ion_index
    )

    scale, max_o_distance_raw = detect_coordinate_scale(
        u,
        ion_index=ion_index,
        water_oxygen_indices=water_oxygen_indices,
        auto_scale=auto_coord_scale,
        user_scale=coord_scale,
    )

    r_values = np.arange(0.0, r_max + 0.5 * dr, dr)
    n_r = len(r_values)

    phi_sum_without_coulomb = np.zeros(n_r, dtype=float)
    n_frames = 0

    trajectory_slice = u.trajectory[start:stop:stride]

    for ts in trajectory_slice:
        ion_pos = u.atoms[ion_index].position.astype(float) * scale
        water_pos = u.atoms[water_indices].positions.astype(float) * scale

        distances = np.linalg.norm(water_pos - ion_pos, axis=1)

        valid = (distances > 1.0e-12) & (distances <= r_max)
        if np.any(valid):
            d = distances[valid]
            q = water_charges[valid]

            contribution = q / d

            # For threshold r_j = j * dr,
            # atom contributes to all r_j >= distance.
            bin_index = np.ceil(d / dr).astype(int)
            bin_index = np.clip(bin_index, 0, n_r - 1)

            hist = np.zeros(n_r, dtype=float)
            np.add.at(hist, bin_index, contribution)

            cumulative = np.cumsum(hist)
            phi_sum_without_coulomb += cumulative

        n_frames += 1

        if progress_every > 0 and n_frames % progress_every == 0:
            print(
                f"    {dcd_path.parent.name}: processed {n_frames} frames",
                flush=True,
            )

    if n_frames == 0:
        raise RuntimeError(f"No frames read from {dcd_path}")

    phi_avg = COULOMB_KCAL_MOL_A_PER_E2 * phi_sum_without_coulomb / n_frames

    return {
        "r_values": r_values,
        "phi_values": phi_avg,
        "n_frames": n_frames,
        "ion_index": ion_index,
        "n_atoms": len(u.atoms),
        "n_water_atoms": len(water_indices),
        "n_water_oxygens": len(water_oxygen_indices),
        "coord_scale": scale,
        "max_o_distance_raw": max_o_distance_raw,
    }


def write_single_charge_csv(
    out_path: Path,
    r_values: np.ndarray,
    phi_values: np.ndarray,
    metadata: Dict,
):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "r_A",
                "Phi_water_to_ion_kcal_mol_per_e",
                "n_frames",
                "coord_scale",
                "ion_index",
                "n_atoms",
                "n_water_atoms",
                "n_water_oxygens",
            ]
        )
        for r, phi in zip(r_values, phi_values):
            writer.writerow(
                [
                    f"{r:.6f}",
                    f"{phi:.12f}",
                    metadata["n_frames"],
                    metadata["coord_scale"],
                    metadata["ion_index"],
                    metadata["n_atoms"],
                    metadata["n_water_atoms"],
                    metadata["n_water_oxygens"],
                ]
            )


def append_combined_rows(
    rows: List[Dict],
    system_name: str,
    ion_label: str,
    q_label: str,
    charge_e: float,
    r_values: np.ndarray,
    phi_values: np.ndarray,
    metadata: Dict,
):
    for r, phi in zip(r_values, phi_values):
        rows.append(
            {
                "system": system_name,
                "ion": ion_label,
                "q_label": q_label,
                "charge_e": charge_e,
                "r_A": r,
                "Phi_water_to_ion_kcal_mol_per_e": phi,
                "n_frames": metadata["n_frames"],
                "coord_scale": metadata["coord_scale"],
                "ion_index": metadata["ion_index"],
                "n_atoms": metadata["n_atoms"],
                "n_water_atoms": metadata["n_water_atoms"],
                "n_water_oxygens": metadata["n_water_oxygens"],
                "max_o_distance_raw": metadata["max_o_distance_raw"],
            }
        )


def write_combined_csv(out_path: Path, rows: List[Dict]):
    fieldnames = [
        "system",
        "ion",
        "q_label",
        "charge_e",
        "r_A",
        "Phi_water_to_ion_kcal_mol_per_e",
        "n_frames",
        "coord_scale",
        "ion_index",
        "n_atoms",
        "n_water_atoms",
        "n_water_oxygens",
        "max_o_distance_raw",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["charge_e"] = f"{out['charge_e']:.10f}"
            out["r_A"] = f"{out['r_A']:.6f}"
            out["Phi_water_to_ion_kcal_mol_per_e"] = (
                f"{out['Phi_water_to_ion_kcal_mol_per_e']:.12f}"
            )
            out["max_o_distance_raw"] = f"{out['max_o_distance_raw']:.6f}"
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate basic water-to-ion potential from single-ion DCD trajectories."
    )

    parser.add_argument(
        "--md_root",
        required=True,
        help="Input MD root folder, e.g. ../../2_MD",
    )
    parser.add_argument(
        "--cluster_dir",
        required=True,
        help="Input cluster PDB folder, e.g. ../../1_prepare/clusters",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output folder",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["1_K", "2_Cl", "3_Mg"],
        choices=["1_K", "2_Cl", "3_Mg"],
        help="Systems to process",
    )
    parser.add_argument(
        "--r_max",
        type=float,
        default=15.0,
        help="Maximum radius in Angstrom",
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=0.1,
        help="Radius spacing in Angstrom",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Read every N frames",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First frame index to read",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop frame index",
    )
    parser.add_argument(
        "--coord_scale",
        type=float,
        default=None,
        help="Coordinate scale factor. Usually leave empty. Use 10 if DCD is read in nm.",
    )
    parser.add_argument(
        "--no_auto_coord_scale",
        action="store_true",
        help="Disable automatic coordinate scale detection",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10000,
        help="Print progress every N frames. Use 0 to disable.",
    )

    args = parser.parse_args()

    md_root = Path(args.md_root).resolve()
    cluster_dir = Path(args.cluster_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for system_name in args.systems:
        info = SYSTEMS[system_name]
        ion_label = info["ion_label"]

        pdb_path = cluster_dir / info["pdb"]
        if not pdb_path.is_file():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        system_md_dir = md_root / system_name / "MD"
        if not system_md_dir.is_dir():
            # fallback, in case user passes md_root/1_K directly
            system_md_dir = md_root / system_name

        if not system_md_dir.is_dir():
            raise FileNotFoundError(f"MD system folder not found: {system_md_dir}")

        charge_dirs = find_charge_dirs(system_md_dir)

        system_out_dir = out_dir / system_name
        system_out_dir.mkdir(parents=True, exist_ok=True)

        system_rows = []

        print(f"\n=== Processing {system_name} ({ion_label}) ===")
        print(f"Topology: {pdb_path}")
        print(f"MD folder: {system_md_dir}")
        print(f"Output folder: {system_out_dir}")
        print(f"Charge folders: {' '.join([p.name for p in charge_dirs])}")

        for q_dir in charge_dirs:
            q_label = q_dir.name
            charge_e = parse_q_label(q_label)
            dcd_path = q_dir / "traj.dcd"

            print(f"\n  Reading {q_label}: {dcd_path}", flush=True)

            result = calculate_potential_for_dcd(
                pdb_path=pdb_path,
                dcd_path=dcd_path,
                system_info=info,
                r_max=args.r_max,
                dr=args.dr,
                stride=args.stride,
                start=args.start,
                stop=args.stop,
                auto_coord_scale=(not args.no_auto_coord_scale),
                coord_scale=args.coord_scale,
                progress_every=args.progress_every,
            )

            r_values = result["r_values"]
            phi_values = result["phi_values"]

            per_charge_csv = system_out_dir / f"{q_label}_potential.csv"
            write_single_charge_csv(
                per_charge_csv,
                r_values,
                phi_values,
                result,
            )

            append_combined_rows(
                system_rows,
                system_name,
                ion_label,
                q_label,
                charge_e,
                r_values,
                phi_values,
                result,
            )

            print(
                f"  Done {q_label}: frames={result['n_frames']}, "
                f"atoms={result['n_atoms']}, water_atoms={result['n_water_atoms']}, "
                f"coord_scale={result['coord_scale']}, "
                f"raw_max_O_distance={result['max_o_distance_raw']:.3f}",
                flush=True,
            )

        system_combined_csv = system_out_dir / "potential_all_charges.csv"
        write_combined_csv(system_combined_csv, system_rows)

        all_rows.extend(system_rows)

        print(f"\nWritten: {system_combined_csv}")

    all_csv = out_dir / "potential_all_systems.csv"
    write_combined_csv(all_csv, all_rows)

    print("\nAll done.")
    print(f"Final combined output: {all_csv}")


if __name__ == "__main__":
    main()

