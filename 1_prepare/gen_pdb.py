#!/usr/bin/env python3
"""
Generate CHARMM modified TIP3P ion-water cluster PDB files.

This script prepares initial PDB files only.
It does not run minimization, equilibration, production MD, or restraints.

Design used here
----------------
Force field:
    CHARMM force field in OpenMM XML format.
    Default tries:
        charmm36_2024.xml + charmm36_2024/water.xml
    or:
        charmm36.xml + charmm36/water.xml

Water model:
    CHARMM modified TIP3P from CHARMM water.xml.

Preparation radius:
    Waters are first selected from oxygen distance < 21 Å.
    This is the PDB preparation shell, not the MD spherical restraint radius.

Water number priority:
    single ion: 1118 waters
    ion pair:   1117 waters
    If more waters than needed are inside the preparation shell, the farthest
    waters from the ion center are removed first.

Ion placement:
    single ion:
        K+, Cl-, Mg2+ at (0, 0, 0)
    ion pair:
        K+/K+, Cl-/Cl-, K+/Cl- separated by d = 1..8 Å
        ion1 at (-d/2, 0, 0), ion2 at (+d/2, 0, 0)

Output:
    PDB files and summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

try:
    from openmm import Vec3
    from openmm.unit import angstrom, nanometer, molar
    from openmm.app import ForceField, Modeller, PDBFile, Topology, element
except ImportError:
    from simtk.openmm import Vec3
    from simtk.unit import angstrom, nanometer, molar
    from simtk.openmm.app import ForceField, Modeller, PDBFile, Topology, element


@dataclass(frozen=True)
class IonDef:
    label: str
    residue_name: str
    atom_name: str
    template_name: str
    elem: object


# CHARMM ion residue/atom names commonly used in OpenMM CHARMM files.
# If your local CHARMM XML uses a different Mg2+ template, edit only this table.
ION_DEFS: Dict[str, IonDef] = {
    "K": IonDef("K", "POT", "POT", "POT", element.potassium),
    "Cl": IonDef("Cl", "CLA", "CLA", "CLA", element.chlorine),
    "Mg": IonDef("Mg", "MG", "MG", "MG", element.magnesium),
}

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL"}
OXYGEN_NAMES = {"O", "OH2", "OW", "OT"}


def load_charmm_forcefield(forcefield_files: Sequence[str] | None = None):
    """Load CHARMM force-field files, preferring CHARMM36_2024 when available."""
    if forcefield_files:
        return ForceField(*forcefield_files), tuple(forcefield_files)

    candidates = [
        ("charmm36_2024.xml", "charmm36_2024/water.xml"),
        ("charmm36.xml", "charmm36/water.xml"),
    ]

    errors = []
    for files in candidates:
        try:
            return ForceField(*files), files
        except Exception as exc:
            errors.append(f"{files}: {exc}")

    raise RuntimeError(
        "Cannot load CHARMM force-field files. Tried:\n  "
        + "\n  ".join(errors)
        + "\nYou may pass explicit files with --ff charmm36.xml charmm36/water.xml"
    )


def make_ion_topology(
    ions: Sequence[str],
    coords_a: Sequence[Tuple[float, float, float]],
):
    """Create an OpenMM Topology for monoatomic CHARMM ions."""
    if len(ions) != len(coords_a):
        raise ValueError("ions and coords_a must have the same length")

    top = Topology()
    chain = top.addChain("A")
    positions = []
    residue_templates = {}

    for i, (ion_label, xyz) in enumerate(zip(ions, coords_a), start=1):
        if ion_label not in ION_DEFS:
            raise ValueError(f"Unknown ion label: {ion_label}. Allowed: {sorted(ION_DEFS)}")

        idef = ION_DEFS[ion_label]
        residue = top.addResidue(idef.residue_name, chain, id=str(i))
        top.addAtom(idef.atom_name, idef.elem, residue, id=str(i))

        # Important: append plain Vec3 objects first, then multiply the whole
        # positions list by the unit. Do not append Vec3(...)*angstrom.
        positions.append(Vec3(float(xyz[0]), float(xyz[1]), float(xyz[2])))

        residue_templates[residue] = idef.template_name

    return top, positions * angstrom, residue_templates


def pair_coords(distance_a: float):
    """Coordinates for a two-ion pair centered at the origin along x."""
    half = float(distance_a) / 2.0
    return [(-half, 0.0, 0.0), (half, 0.0, 0.0)]


def single_coords():
    """Coordinate for a single ion at the origin."""
    return [(0.0, 0.0, 0.0)]


def pos_to_tuple_a(pos):
    """Return an OpenMM position as a numeric tuple in Å."""
    v = pos.value_in_unit(angstrom)
    return (float(v[0]), float(v[1]), float(v[2]))


def distance_a(p, q):
    return math.sqrt(
        (p[0] - q[0]) ** 2
        + (p[1] - q[1]) ** 2
        + (p[2] - q[2]) ** 2
    )


def is_water_residue(residue):
    """Detect water residues by residue name and atom composition."""
    atoms = list(residue.atoms())

    if residue.name in WATER_RESNAMES:
        return True

    n_o = 0
    n_h = 0

    for atom in atoms:
        if atom.element == element.oxygen or atom.name in OXYGEN_NAMES:
            n_o += 1
        elif atom.element == element.hydrogen or atom.name.upper().startswith("H"):
            n_h += 1

    return len(atoms) == 3 and n_o == 1 and n_h == 2


def water_oxygen_atom(residue):
    atoms = list(residue.atoms())
    oxygen_atoms = [
        atom for atom in atoms
        if atom.element == element.oxygen or atom.name in OXYGEN_NAMES
    ]

    if len(oxygen_atoms) != 1:
        raise RuntimeError(
            f"Could not identify a unique water oxygen in residue "
            f"{residue.name} {residue.id}; atoms={[a.name for a in atoms]}"
        )

    return oxygen_atoms[0]


def ion_atoms_from_topology(topology):
    """Return atoms belonging to residues named as the requested CHARMM ions."""
    ion_resnames = {d.residue_name for d in ION_DEFS.values()}
    return [
        atom for atom in topology.atoms()
        if atom.residue.name in ion_resnames
    ]


def center_from_ion_atoms(topology, positions):
    """Use average ion position as cluster center."""
    ions = ion_atoms_from_topology(topology)

    if not ions:
        raise RuntimeError("No ion atoms found when computing cluster center.")

    coords = [pos_to_tuple_a(positions[atom.index]) for atom in ions]
    n = float(len(coords))

    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )


def recentered_positions(positions, center_a):
    """
    Return one Quantity containing a list of plain Vec3 objects.

    Do not return a list where each atom position is separately multiplied by unit.
    """
    centered = []

    for p in positions:
        v = p.value_in_unit(angstrom)
        centered.append(
            Vec3(
                float(v[0]) - center_a[0],
                float(v[1]) - center_a[1],
                float(v[2]) - center_a[2],
            )
        )

    return centered * angstrom


def count_waters(topology):
    return sum(1 for res in topology.residues() if is_water_residue(res))


def count_ions(topology):
    return len(ion_atoms_from_topology(topology))


def water_o_radii_a(topology, positions, center_a):
    radii = []

    for res in topology.residues():
        if is_water_residue(res):
            o_atom = water_oxygen_atom(res)
            radii.append(distance_a(pos_to_tuple_a(positions[o_atom.index]), center_a))

    return radii


def min_ion_water_o_distance_a(topology, positions):
    ion_atoms = ion_atoms_from_topology(topology)
    water_o_atoms = [
        water_oxygen_atom(res)
        for res in topology.residues()
        if is_water_residue(res)
    ]

    if not ion_atoms or not water_o_atoms:
        return None

    min_d = None

    for ion in ion_atoms:
        p = pos_to_tuple_a(positions[ion.index])

        for ow in water_o_atoms:
            q = pos_to_tuple_a(positions[ow.index])
            d = distance_a(p, q)

            if min_d is None or d < min_d:
                min_d = d

    return min_d


def select_cluster_waters(modeller, target_waters: int, selection_radius_a: float):
    """
    Keep exactly target_waters water molecules.

    Preparation rule:
    1. Use waters with oxygen distance from ion center < selection_radius_a.
    2. Fixed water number has priority.
    3. If too many waters are available, remove farthest waters first.

    selection_radius_a is for initial PDB preparation only.
    It is not the later MD spherical restraint radius.
    """
    topology = modeller.topology
    positions = modeller.positions
    center_a = center_from_ion_atoms(topology, positions)

    water_records = []
    all_water_residues = []

    for res in topology.residues():
        if not is_water_residue(res):
            continue

        all_water_residues.append(res)
        o_atom = water_oxygen_atom(res)
        r = distance_a(pos_to_tuple_a(positions[o_atom.index]), center_a)
        water_records.append((r, res))

    candidates = [
        (r, res) for (r, res) in water_records
        if r < selection_radius_a
    ]

    if len(candidates) < target_waters:
        raise RuntimeError(
            f"Only {len(candidates)} waters found with oxygen radius "
            f"< {selection_radius_a:.3f} Å, but target is {target_waters}. "
            f"Increase --selection_radius_a or --box_a."
        )

    # Fixed water-number priority:
    # Sort by water oxygen distance from ion center and keep closest target_waters.
    candidates_sorted = sorted(candidates, key=lambda x: x[0])
    selected = candidates_sorted[:target_waters]
    selected_residues = {res for _, res in selected}

    residues_to_delete = [
        res for res in all_water_residues
        if res not in selected_residues
    ]

    if residues_to_delete:
        modeller.delete(residues_to_delete)


def generate_one_case(
    case_name: str,
    ions: Sequence[str],
    coords_a: Sequence[Tuple[float, float, float]],
    target_waters: int,
    forcefield,
    out_dir: Path,
    box_a: float = 45.0,
    selection_radius_a: float = 21.0,
    clear_box_vectors: bool = True,
):
    top, positions, residue_templates = make_ion_topology(ions, coords_a)
    modeller = Modeller(top, positions)

    modeller.addSolvent(
        forcefield,
        model="tip3p",
        boxSize=Vec3(box_a / 10.0, box_a / 10.0, box_a / 10.0) * nanometer,
        neutralize=False,
        ionicStrength=0 * molar,
        residueTemplates=residue_templates,
    )

    select_cluster_waters(
        modeller,
        target_waters=target_waters,
        selection_radius_a=selection_radius_a,
    )

    final_top = modeller.topology
    final_pos = modeller.positions

    center_a = center_from_ion_atoms(final_top, final_pos)
    final_pos_centered = recentered_positions(final_pos, center_a)

    if clear_box_vectors:
        try:
            final_top.setPeriodicBoxVectors(None)
        except Exception:
            # Older OpenMM versions may not accept None.
            # This is not fatal for PDB output.
            pass

    out_path = out_dir / f"{case_name}.pdb"

    with out_path.open("w") as f:
        PDBFile.writeFile(final_top, final_pos_centered, f, keepIds=True)

    zero = (0.0, 0.0, 0.0)
    radii = water_o_radii_a(final_top, final_pos_centered, zero)
    min_iw = min_ion_water_o_distance_a(final_top, final_pos_centered)

    return {
        "case": case_name,
        "ions": "+".join(ions),
        "target_waters": target_waters,
        "actual_waters": count_waters(final_top),
        "actual_ions": count_ions(final_top),
        "total_atoms": sum(1 for _ in final_top.atoms()),
        "min_water_O_radius_A": min(radii) if radii else None,
        "max_water_O_radius_A": max(radii) if radii else None,
        "min_ion_water_O_distance_A": min_iw,
        "output": str(out_path),
    }


def format_float(x):
    if x is None:
        return "NA"

    if isinstance(x, float):
        return f"{x:.4f}"

    return str(x)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate CHARMM TIP3P ion-water cluster PDB files "
            "with fixed water counts."
        )
    )

    parser.add_argument(
        "--out_dir",
        default="charmm_tip3p_clusters",
        help="Output directory",
    )

    parser.add_argument(
        "--box_a",
        type=float,
        default=45.0,
        help="Initial cubic water box length in Å",
    )

    parser.add_argument(
        "--selection_radius_a",
        type=float,
        default=21.0,
        help=(
            "Preparation water-selection radius in Å, based on water oxygen. "
            "This is not the MD spherical restraint radius."
        ),
    )

    parser.add_argument(
        "--single_waters",
        type=int,
        default=1118,
        help="Number of waters for single-ion cases",
    )

    parser.add_argument(
        "--pair_waters",
        type=int,
        default=1117,
        help="Number of waters for ion-pair cases",
    )

    parser.add_argument(
        "--ff",
        nargs="+",
        default=None,
        help=(
            "Explicit OpenMM CHARMM XML files, e.g. "
            "--ff charmm36.xml charmm36/water.xml"
        ),
    )

    parser.add_argument(
        "--keep_box_vectors",
        action="store_true",
        help=(
            "Keep the 45 Å periodic box vectors in the PDB CRYST1 record. "
            "Default tries to remove them."
        ),
    )

    parser.add_argument(
        "--skip_single",
        action="store_true",
        help="Do not generate single-ion cases.",
    )

    parser.add_argument(
        "--skip_pairs",
        action="store_true",
        help="Do not generate ion-pair cases.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forcefield, ff_files = load_charmm_forcefield(args.ff)

    print("Loaded CHARMM force field files:", ", ".join(ff_files))
    print("This script prepares PDB only; it does not run MD.")
    print(f"Preparation selection radius: O distance < {args.selection_radius_a:.3f} Å")
    print("Later MD spherical restraint radius is not applied here.\n")

    records = []

    if not args.skip_single:
        for ion in ["K", "Cl", "Mg"]:
            case = f"single_{ion}_sel{int(args.selection_radius_a)}A_{args.single_waters}w"

            rec = generate_one_case(
                case_name=case,
                ions=[ion],
                coords_a=single_coords(),
                target_waters=args.single_waters,
                forcefield=forcefield,
                out_dir=out_dir,
                box_a=args.box_a,
                selection_radius_a=args.selection_radius_a,
                clear_box_vectors=not args.keep_box_vectors,
            )

            records.append(rec)

            print(
                f"{rec['case']:36s} "
                f"waters={rec['actual_waters']:4d} "
                f"ions={rec['actual_ions']} "
                f"max_O_r={format_float(rec['max_water_O_radius_A'])} Å "
                f"min_ion_O={format_float(rec['min_ion_water_O_distance_A'])} Å"
            )

    if not args.skip_pairs:
        pair_list = [
            ("K", "K"),
            ("Cl", "Cl"),
            ("K", "Cl"),
        ]

        for ion1, ion2 in pair_list:
            for d in range(1, 9):
                case = (
                    f"pair_{ion1}_{ion2}_d{d:02d}A_"
                    f"sel{int(args.selection_radius_a)}A_{args.pair_waters}w"
                )

                rec = generate_one_case(
                    case_name=case,
                    ions=[ion1, ion2],
                    coords_a=pair_coords(float(d)),
                    target_waters=args.pair_waters,
                    forcefield=forcefield,
                    out_dir=out_dir,
                    box_a=args.box_a,
                    selection_radius_a=args.selection_radius_a,
                    clear_box_vectors=not args.keep_box_vectors,
                )

                records.append(rec)

                print(
                    f"{rec['case']:36s} "
                    f"waters={rec['actual_waters']:4d} "
                    f"ions={rec['actual_ions']} "
                    f"max_O_r={format_float(rec['max_water_O_radius_A'])} Å "
                    f"min_ion_O={format_float(rec['min_ion_water_O_distance_A'])} Å"
                )

    summary_path = out_dir / "summary.csv"

    if records:
        fieldnames = list(records[0].keys())

        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    print(f"\nDone. Generated {len(records)} PDB files.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
