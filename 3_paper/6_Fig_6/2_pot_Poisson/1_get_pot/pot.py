#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict


def mean(values):
    return sum(values) / len(values)


def fmt_value(x, ndigits=1):
    if x is None:
        return ""
    return round(x, ndigits)


def main():
    parser = argparse.ArgumentParser(
        description="Convert APBS two center output into a wide summary table."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV, for example two_center_apbs_reaction_potentials.csv",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV, for example two_center_apbs_summary.csv",
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=1,
        help="Number of decimal places. Default: 1",
    )

    args = parser.parse_args()

    kk_by_d = defaultdict(list)
    clcl_by_d = defaultdict(list)
    kcl_k_by_d = {}
    kcl_cl_by_d = {}

    all_d = set()

    with open(args.input, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            pair = row["pair"]
            d = int(float(row["d_angstrom"]))
            atom_index = int(row["atom_index"])
            atom_name = row["atom_name"]
            value = float(row["reaction_kcal_per_mol_e"])

            all_d.add(d)

            if pair == "K_K":
                kk_by_d[d].append(value)

            elif pair == "CL_CL":
                clcl_by_d[d].append(value)

            elif pair == "K_CL":
                if atom_index == 1 and atom_name == "K":
                    kcl_k_by_d[d] = value
                elif atom_index == 2 and atom_name == "CL":
                    kcl_cl_by_d[d] = value
                else:
                    raise ValueError(
                        f"Unexpected K_CL atom assignment: "
                        f"d={d}, atom_index={atom_index}, atom_name={atom_name}"
                    )

            else:
                raise ValueError(f"Unknown pair name: {pair}")

    rows_out = []

    for d in sorted(all_d):
        kk_avg = mean(kk_by_d[d]) if kk_by_d[d] else None
        clcl_avg = mean(clcl_by_d[d]) if clcl_by_d[d] else None
        kcl_k = kcl_k_by_d.get(d)
        kcl_cl = kcl_cl_by_d.get(d)

        rows_out.append({
            "Dis": d,
            "K1(KK)": fmt_value(kk_avg, args.digits),
            "CL1(CLCL)": fmt_value(clcl_avg, args.digits),
            "K1(KCL)": fmt_value(kcl_k, args.digits),
            "CL2(KCL)": fmt_value(kcl_cl, args.digits),
        })

    fieldnames = [
        "Dis",
        "K1(KK)",
        "CL1(CLCL)",
        "K1(KCL)",
        "CL2(KCL)",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Done. Output written to: {args.output}")


if __name__ == "__main__":
    main()

