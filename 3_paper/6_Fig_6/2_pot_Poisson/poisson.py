#!/usr/bin/env python3

from pathlib import Path
import argparse
import subprocess
import csv
import shutil
import os


# ============================================================
# Parameters
# ============================================================

PARAMS = {
    "K": {
        "charge": 1.0,
        "radius": 1.886,
    },
    "CL": {
        "charge": -1.0,
        "radius": 1.773,
    },
}

PAIRS = [
    ("K_K", "K", "K"),
    ("CL_CL", "CL", "CL"),
    ("K_CL", "K", "CL"),
]


# ============================================================
# File writers
# ============================================================

def write_pqr(path, atom1, atom2, d):
    p1 = PARAMS[atom1]
    p2 = PARAMS[atom2]

    x1 = -d / 2.0
    x2 = d / 2.0

    lines = [
        f"ATOM      1 {atom1:<4s} MOD A   1    "
        f"{x1:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{p1['charge']:8.4f}{p1['radius']:8.4f}\n",

        f"ATOM      2 {atom2:<4s} MOD A   1    "
        f"{x2:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{p2['charge']:8.4f}{p2['radius']:8.4f}\n",

        "END\n",
    ]

    Path(path).write_text("".join(lines))


def write_apbs_input(path, pqr_file, sdie, output_prefix):
    text = f"""
read
    mol pqr {pqr_file}
end

elec
    mg-auto
    dime 321 321 321
    cglen 50.0 50.0 50.0
    fglen 18.0 18.0 18.0
    cgcent mol 1
    fgcent mol 1

    mol 1
    lpbe
    bcfl sdh
    pdie 1.0
    sdie {sdie}
    srfm mol
    srad 0.0
    chgm spl2
    sdens 10.0
    temp 300.0

    write atompot flat {output_prefix}
end

quit
"""
    Path(path).write_text(text.strip() + "\n")


# ============================================================
# APBS runner and output reader
# ============================================================

def run_apbs(apbs_cmd, input_file, log_file):
    with open(log_file, "w") as log:
        subprocess.run(
            [apbs_cmd, input_file],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def find_atompot_file(prefix):
    candidates = [
        f"{prefix}.txt",
        f"{prefix}.flat",
        f"{prefix}.dat",
        prefix,
    ]

    for c in candidates:
        if Path(c).exists():
            return Path(c)

    matches = sorted(Path(".").glob(f"{prefix}*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Cannot find APBS atom potential file for prefix: {prefix}"
    )


def read_atompot(prefix, expected_n=2):
    """
    APBS atompot output may contain extra numeric values before the atom potentials.
    Example observed:
        1.5
        6055.799
        6055.799

    Since this system always has two charge centers, the actual atom potentials
    are taken as the last two numeric values.
    """
    file_path = find_atompot_file(prefix)
    values = []

    for line in file_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        for part in line.split():
            try:
                values.append(float(part))
            except ValueError:
                pass

    if len(values) < expected_n:
        raise ValueError(
            f"Expected at least {expected_n} numeric values in {file_path}, "
            f"but got {len(values)}. Parsed values: {values}"
        )

    return values[-expected_n:]


# ============================================================
# One APBS case
# ============================================================

def run_one_case(pair_name, atom1, atom2, d, out_dir, apbs_cmd, rt_kcal):
    workdir = out_dir / pair_name / f"d{int(d):02d}"
    workdir.mkdir(parents=True, exist_ok=True)

    old_dir = Path.cwd()

    try:
        os.chdir(workdir)

        write_pqr("two_centers.pqr", atom1, atom2, float(d))

        # Solvated state: solvent dielectric = 80
        write_apbs_input(
            path="solv.in",
            pqr_file="two_centers.pqr",
            sdie=80.0,
            output_prefix="solv_atompot",
        )
        run_apbs(apbs_cmd, "solv.in", "solv.log")

        # Reference state: dielectric = 1
        write_apbs_input(
            path="ref.in",
            pqr_file="two_centers.pqr",
            sdie=1.0,
            output_prefix="ref_atompot",
        )
        run_apbs(apbs_cmd, "ref.in", "ref.log")

        phi_solv = read_atompot("solv_atompot", expected_n=2)
        phi_ref = read_atompot("ref_atompot", expected_n=2)

        rows = []

        for atom_index, atom_name in enumerate([atom1, atom2], start=1):
            ps = phi_solv[atom_index - 1]
            pr = phi_ref[atom_index - 1]

            reaction_kbt_per_e = ps - pr
            reaction_kcal_per_mol_e = reaction_kbt_per_e * rt_kcal

            rows.append({
                "pair": pair_name,
                "d_angstrom": d,
                "atom_index": atom_index,
                "atom_name": atom_name,
                "charge_e": PARAMS[atom_name]["charge"],
                "radius_angstrom": PARAMS[atom_name]["radius"],
                "phi_solv_kBT_per_e": ps,
                "phi_ref_kBT_per_e": pr,
                "reaction_kBT_per_e": reaction_kbt_per_e,
                "reaction_kcal_per_mol_e": reaction_kcal_per_mol_e,
            })

        return rows

    finally:
        os.chdir(old_dir)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run APBS Poisson calculations for two charge centers."
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--apbs",
        default="apbs",
        help="APBS executable name or path. Default: apbs",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in K. Default: 300",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before running.",
    )

    args = parser.parse_args()

    if shutil.which(args.apbs) is None:
        raise RuntimeError(
            f"Cannot find APBS executable: {args.apbs}\n"
            "Install APBS first, for example:\n"
            "conda install -c conda-forge apbs"
        )

    out_dir = Path(args.out_dir).resolve()

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    rt_kcal = 0.00198720425864083 * args.temperature

    all_rows = []

    for pair_name, atom1, atom2 in PAIRS:
        for d in range(1, 9):
            print(f"Running {pair_name}, d = {d} Å")

            rows = run_one_case(
                pair_name=pair_name,
                atom1=atom1,
                atom2=atom2,
                d=d,
                out_dir=out_dir,
                apbs_cmd=args.apbs,
                rt_kcal=rt_kcal,
            )

            all_rows.extend(rows)

    out_csv = out_dir / "two_center_apbs_reaction_potentials.csv"

    fieldnames = [
        "pair",
        "d_angstrom",
        "atom_index",
        "atom_name",
        "charge_e",
        "radius_angstrom",
        "phi_solv_kBT_per_e",
        "phi_ref_kBT_per_e",
        "reaction_kBT_per_e",
        "reaction_kcal_per_mol_e",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("Done.")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()

