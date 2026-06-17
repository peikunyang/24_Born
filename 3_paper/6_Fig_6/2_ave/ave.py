#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SYSTEMS = {
    "4_K_K": {
        "full": "4_K_K_cumulative_potential.csv",
        "q0": "4_K_K_Q0_cumulative_potential.csv",
        "delta": "4_K_K_delta_cumulative_potential.csv",
        "avg_cols": ("K1", "K2"),
        "summary_name": "K1(KK)",
    },
    "5_CL_CL": {
        "full": "5_CL_CL_cumulative_potential.csv",
        "q0": "5_CL_CL_Q0_cumulative_potential.csv",
        "delta": "5_CL_CL_delta_cumulative_potential.csv",
        "avg_cols": ("CL1", "CL2"),
        "summary_name": "CL1(CLCL)",
    },
    "6_K_CL": {
        "full": "6_K_CL_cumulative_potential.csv",
        "q0": "6_K_CL_Q0_cumulative_potential.csv",
        "delta": "6_K_CL_delta_cumulative_potential.csv",
        "avg_cols": None,
        "summary_name": None,
    },
}


def read_csv_data(path):
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames is None:
        raise RuntimeError(f"Empty CSV: {path}")

    if "r" not in fieldnames:
        raise RuntimeError(f"No r column in {path}")

    r_values = [float(row["r"]) for row in rows]

    data = {}
    for col in fieldnames:
        if col == "r":
            continue
        data[col] = [float(row[col]) for row in rows]

    return r_values, data, fieldnames


def check_r_grid(r1, r2, name):
    if len(r1) != len(r2):
        raise RuntimeError(f"r grid length mismatch: {name}")

    for a, b in zip(r1, r2):
        if abs(a - b) > 1e-8:
            raise RuntimeError(f"r grid mismatch: {name}")


def subtract_full_minus_q0(r_full, full_data, r_q0, q0_data, system_name):
    check_r_grid(r_full, r_q0, system_name)

    delta = {}

    for col in full_data:
        if col not in q0_data:
            raise RuntimeError(f"{system_name}: column {col} not found in Q0 file")

        delta[col] = [
            a - b for a, b in zip(full_data[col], q0_data[col])
        ]

    return r_full, delta


def write_delta_csv(path, r_values, delta_data):
    cols = list(delta_data.keys())

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(["r"] + cols)

        for i, r in enumerate(r_values):
            row = [f"{r:g}"]
            for col in cols:
                row.append(f"{delta_data[col][i]:.6f}")
            writer.writerow(row)


def average_10_14(r_values, values, r_min, r_max):
    vals = [
        v for r, v in zip(r_values, values)
        if r_min <= r <= r_max
    ]

    if not vals:
        raise RuntimeError("No data in r = 10–14 Å region")

    return sum(vals) / len(vals)


def avg_two(a, b):
    return 0.5 * (a + b)


def process_system(system_name, info, in_dir, out_dir):
    full_path = in_dir / info["full"]
    q0_path = in_dir / info["q0"]

    if not full_path.is_file():
        raise FileNotFoundError(full_path)

    if not q0_path.is_file():
        raise FileNotFoundError(q0_path)

    r_full, full_data, _ = read_csv_data(full_path)
    r_q0, q0_data, _ = read_csv_data(q0_path)

    r_values, delta_data = subtract_full_minus_q0(
        r_full,
        full_data,
        r_q0,
        q0_data,
        system_name,
    )

    out_path = out_dir / info["delta"]
    write_delta_csv(out_path, r_values, delta_data)

    print(f"written: {out_path}")

    return r_values, delta_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--r_min", type=float, default=10.0)
    parser.add_argument("--r_max", type=float, default=14.0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for system_name, info in SYSTEMS.items():
        r_values, delta_data = process_system(
            system_name=system_name,
            info=info,
            in_dir=in_dir,
            out_dir=out_dir,
        )
        results[system_name] = {
            "r": r_values,
            "data": delta_data,
        }

    rows = []
    rows.append([
        "Dis",
        "K1(KK)",
        "CL1(CLCL)",
        "K1(KCL)",
        "CL2(KCL)",
    ])

    for d in range(1, 9):
        dlab = f"d{d:02d}"

        r_kk = results["4_K_K"]["r"]
        kk = results["4_K_K"]["data"]

        kk_1 = average_10_14(
            r_kk,
            kk[f"{dlab}_K1"],
            args.r_min,
            args.r_max,
        )
        kk_2 = average_10_14(
            r_kk,
            kk[f"{dlab}_K2"],
            args.r_min,
            args.r_max,
        )
        kk_avg = avg_two(kk_1, kk_2)

        r_clcl = results["5_CL_CL"]["r"]
        clcl = results["5_CL_CL"]["data"]

        clcl_1 = average_10_14(
            r_clcl,
            clcl[f"{dlab}_CL1"],
            args.r_min,
            args.r_max,
        )
        clcl_2 = average_10_14(
            r_clcl,
            clcl[f"{dlab}_CL2"],
            args.r_min,
            args.r_max,
        )
        clcl_avg = avg_two(clcl_1, clcl_2)

        r_kcl = results["6_K_CL"]["r"]
        kcl = results["6_K_CL"]["data"]

        kcl_k = average_10_14(
            r_kcl,
            kcl[f"{dlab}_K"],
            args.r_min,
            args.r_max,
        )
        kcl_cl = average_10_14(
            r_kcl,
            kcl[f"{dlab}_CL"],
            args.r_min,
            args.r_max,
        )

        rows.append([
            d,
            f"{kk_avg:.1f}",
            f"{clcl_avg:.1f}",
            f"{kcl_k:.1f}",
            f"{kcl_cl:.1f}",
        ])

    out_csv = Path(args.out_csv)

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"written: {out_csv}")


if __name__ == "__main__":
    main()

