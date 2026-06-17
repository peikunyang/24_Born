#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

SYSTEMS = {
    "1_K": {
        "title": "K",
        "baseline": "q0.0",
        "targets": [
            ("q0.1", "0.1"),
            ("q0.2", "0.2"),
            ("q0.3", "0.3"),
            ("q0.4", "0.4"),
            ("q0.5", "0.5"),
            ("q0.6", "0.6"),
            ("q0.7", "0.7"),
            ("q0.8", "0.8"),
            ("q0.9", "0.9"),
            ("q1.0", "1"),
        ],
    },
    "2_Cl": {
        "title": "CL",
        "baseline": "q0.0",
        "targets": [
            ("qm0.1", "-0.1"),
            ("qm0.2", "-0.2"),
            ("qm0.3", "-0.3"),
            ("qm0.4", "-0.4"),
            ("qm0.5", "-0.5"),
            ("qm0.6", "-0.6"),
            ("qm0.7", "-0.7"),
            ("qm0.8", "-0.8"),
            ("qm0.9", "-0.9"),
            ("qm1.0", "-1"),
        ],
    },
    "3_Mg": {
        "title": "Mg",
        "baseline": "q0.0",
        "targets": [
            ("q0.2", "0.2"),
            ("q0.4", "0.4"),
            ("q0.6", "0.6"),
            ("q0.8", "0.8"),
            ("q1.0", "1"),
            ("q1.2", "1.2"),
            ("q1.4", "1.4"),
            ("q1.6", "1.6"),
            ("q1.8", "1.8"),
            ("q2.0", "2"),
        ],
    },
}


def read_potential(path):
    data = {}

    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = float(row["r_A"])
            phi = float(row["Phi_water_to_ion_kcal_mol_per_e"])
            data[r] = phi

    if not data:
        raise RuntimeError(f"沒有讀到資料: {path}")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pot_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    pot_dir = Path(args.pot_dir)

    blocks = []
    r_ref = None

    for system_name, info in SYSTEMS.items():
        sys_dir = pot_dir / system_name

        base_file = sys_dir / f"{info['baseline']}_potential.csv"
        if not base_file.is_file():
            raise FileNotFoundError(f"找不到 baseline 檔案: {base_file}")

        base = read_potential(base_file)

        if r_ref is None:
            r_ref = set(base.keys())
        elif r_ref != set(base.keys()):
            raise RuntimeError(f"r_A 不一致: {base_file}")

        delta_curves = []

        for q_label, charge_label in info["targets"]:
            target_file = sys_dir / f"{q_label}_potential.csv"
            if not target_file.is_file():
                raise FileNotFoundError(f"找不到 target 檔案: {target_file}")

            target = read_potential(target_file)

            if set(target.keys()) != r_ref:
                raise RuntimeError(f"r_A 不一致: {target_file}")

            delta = {}
            for r in r_ref:
                delta[r] = target[r] - base[r]

            delta_curves.append((charge_label, delta))

        blocks.append((info["title"], delta_curves))

    r_values = sorted(r_ref)

    rows = []

    header = []
    for i, (title, curves) in enumerate(blocks):
        if i > 0:
            header.append("")
        header.extend([title, "Div"])
        header.extend([charge_label for charge_label, _ in curves])
    rows.append(header)

    for r in r_values:
        row = []

        for i, (title, curves) in enumerate(blocks):
            if i > 0:
                row.append("")

            row.append("")
            row.append(f"{r:g}")

            for charge_label, delta in curves:
                row.append(f"{delta[r]:.2f}")

        rows.append(row)

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":
    main()

