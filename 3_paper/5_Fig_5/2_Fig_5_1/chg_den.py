#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


TARGETS = {
    "K":  ["0.5", "1"],
    "CL": ["-0.5", "-1"],
    "MG": ["1", "2"],
    "Mg": ["1", "2"],
}


def is_number(x):
    try:
        float(str(x).strip())
        return True
    except:
        return False


def fmt_val(x):
    return f"{x:.3f}".rstrip("0").rstrip(".")


def fmt_r(x):
    return f"{x:g}"


def read_raw_chg_den(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    blocks = []

    i = 0
    while i < len(header):
        title = header[i].strip()

        if title in {"K", "CL", "MG", "Mg"}:
            r_col = i + 1
            q_cols = {}

            j = i + 2
            while j < len(header):
                h = header[j].strip()
                if h == "":
                    break
                if is_number(h):
                    q_cols[f"{float(h):g}"] = j
                j += 1

            blocks.append((title, r_col, q_cols))
            i = j + 1
        else:
            i += 1

    data = {}

    for title, r_col, q_cols in blocks:
        data[title] = {}

        for q_label, col in q_cols.items():
            data[title][q_label] = {}

        for row in rows[1:]:
            if r_col >= len(row):
                continue
            if not is_number(row[r_col]):
                continue

            r = float(row[r_col])

            for q_label, col in q_cols.items():
                if col < len(row) and is_number(row[col]):
                    data[title][q_label][r] = float(row[col])

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    data = read_raw_chg_den(Path(args.input_csv))

    titles = ["K", "CL", "MG"]
    if "MG" not in data and "Mg" in data:
        titles = ["K", "CL", "Mg"]

    r_ref = None
    results = {}

    for title in titles:
        if "0" not in data[title]:
            raise RuntimeError(f"{title} 找不到 q=0 欄位")

        base = data[title]["0"]

        if r_ref is None:
            r_ref = set(base.keys())
        elif r_ref != set(base.keys()):
            raise RuntimeError(f"{title} 的 r grid 不一致")

        results[title] = {}

        for q_label in TARGETS[title]:
            if q_label not in data[title]:
                raise RuntimeError(f"{title} 找不到 q={q_label} 欄位")

            target = data[title][q_label]

            if set(target.keys()) != r_ref:
                raise RuntimeError(f"{title} q={q_label} 的 r grid 不一致")

            results[title][q_label] = {
                r: target[r] - base[r]
                for r in r_ref
            }

    r_values = sorted(r_ref)

    rows = []

    header = []
    for title in titles:
        if title == "K":
            header.extend(["", title])
        else:
            header.extend([title])
        header.extend(TARGETS[title])
    header.append("")
    rows.append(header)

    for r in r_values:
        row = []

        row.append(fmt_r(r))
        row.append("")

        for q_label in TARGETS["K"]:
            row.append(fmt_val(results["K"][q_label][r]))

        row.append("")

        for q_label in TARGETS["CL"]:
            row.append(fmt_val(results["CL"][q_label][r]))

        row.append("")

        mg_title = "MG" if "MG" in results else "Mg"
        for q_label in TARGETS[mg_title]:
            row.append(fmt_val(results[mg_title][q_label][r]))

        row.append(fmt_r(r))

        rows.append(row)

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":
    main()

