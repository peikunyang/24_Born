#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import statistics


ION_ORDER = ["K", "CL", "Mg", "MG"]


def is_number(x):
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


def fmt_percent(x):
    return f"{x:.2f}%"


def read_fig1_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise RuntimeError(f"empty file: {path}")

    header = rows[0]
    blocks = []

    i = 0
    while i < len(header):
        title = header[i].strip()

        if title in {"K", "CL", "Mg", "MG"}:
            r_col = i + 1
            q_cols = []

            j = i + 2
            while j < len(header):
                h = header[j].strip()

                if h == "":
                    break

                if h in {"K", "CL", "Mg", "MG"}:
                    break

                if is_number(h):
                    q_cols.append((h, j))

                j += 1

            blocks.append({
                "title": title,
                "r_col": r_col,
                "q_cols": q_cols,
            })

            i = j + 1
        else:
            i += 1

    if not blocks:
        raise RuntimeError("No ion blocks found. Expected headers such as K, Div, 0.2, 0.4 ...")

    data = {}

    for block in blocks:
        title = block["title"]
        r_col = block["r_col"]
        q_cols = block["q_cols"]

        data[title] = {}

        for q_label, col in q_cols:
            vals = []

            for row in rows[1:]:
                if r_col >= len(row) or col >= len(row):
                    continue

                r_txt = row[r_col].strip()
                y_txt = row[col].strip()

                if not is_number(r_txt) or not is_number(y_txt):
                    continue

                r = float(r_txt)
                y = float(y_txt)

                vals.append((r, y))

            data[title][q_label] = vals

    return data


def calc_ratio(values, r_min, r_max, sample_std=False):
    selected = [
        y for r, y in values
        if r_min <= r <= r_max
    ]

    if not selected:
        raise RuntimeError(f"No data in r = {r_min} to {r_max} Å")

    avg = sum(selected) / len(selected)

    if len(selected) == 1:
        std = 0.0
    else:
        if sample_std:
            std = statistics.stdev(selected)
        else:
            std = statistics.pstdev(selected)

    if abs(avg) < 1e-12:
        ratio = None
    else:
        ratio = 100.0 * std / abs(avg)

    return avg, std, ratio, len(selected)


def ion_display_name(title):
    if title == "K":
        return "K+"
    if title == "CL":
        return "Cl−"
    if title in {"Mg", "MG"}:
        return "Mg2+"
    return title


def charge_order(title, q_labels):
    return sorted(q_labels, key=lambda x: abs(float(x)))


def write_output_csv(out_csv, results):
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ion",
            "q_label",
            "mean_10_14",
            "std_10_14",
            "std_over_abs_mean_percent",
            "n_points",
        ])

        for title, rows in results.items():
            for rec in rows:
                writer.writerow([
                    title,
                    rec["q_label"],
                    f"{rec['mean']:.10f}",
                    f"{rec['std']:.10f}",
                    "" if rec["ratio_percent"] is None else f"{rec['ratio_percent']:.6f}",
                    rec["n_points"],
                ])


def print_article_sentence(results):
    for title in ["K", "CL", "Mg", "MG"]:
        if title not in results:
            continue

        ratios = []
        q_labels = []

        for rec in results[title]:
            q_labels.append(rec["q_label"])
            ratios.append(
                "NA" if rec["ratio_percent"] is None
                else fmt_percent(rec["ratio_percent"])
            )

        print()
        print(f"{ion_display_name(title)}:")
        print(", ".join(ratios))
        print("charge labels:")
        print(", ".join(q_labels))


def main():
    parser = argparse.ArgumentParser(
        description="Calculate std/abs(mean) ratios for Fig. 1 data over r = 10 to 14 Å."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--r_min", type=float, default=10.0)
    parser.add_argument("--r_max", type=float, default=14.0)
    parser.add_argument(
        "--sample_std",
        action="store_true",
        help="Use sample standard deviation. Default uses population standard deviation.",
    )
    args = parser.parse_args()

    data = read_fig1_csv(Path(args.input_csv))

    results = {}

    for title in ["K", "CL", "Mg", "MG"]:
        if title not in data:
            continue

        rows = []

        for q_label in charge_order(title, data[title].keys()):
            avg, std, ratio, n_points = calc_ratio(
                data[title][q_label],
                args.r_min,
                args.r_max,
                sample_std=args.sample_std,
            )

            rows.append({
                "q_label": q_label,
                "mean": avg,
                "std": std,
                "ratio_percent": ratio,
                "n_points": n_points,
            })

        results[title] = rows

    write_output_csv(args.out_csv, results)

    print(f"written: {args.out_csv}")
    print_article_sentence(results)


if __name__ == "__main__":
    main()

