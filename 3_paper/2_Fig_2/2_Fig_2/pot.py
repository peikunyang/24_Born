#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def is_number(x):
    try:
        float(str(x).strip())
        return True
    except:
        return False


def read_pot_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    blocks = []

    i = 0
    while i < len(header):
        title = header[i].strip()

        if title in {"K", "CL", "Mg", "MG"}:
            r_col = i + 1
            charge_cols = []

            j = i + 2
            while j < len(header):
                h = header[j].strip()
                if h == "":
                    break
                if is_number(h):
                    charge_cols.append((float(h), j))
                j += 1

            blocks.append({
                "title": title,
                "r_col": r_col,
                "charge_cols": charge_cols,
            })

            i = j + 1
        else:
            i += 1

    data = {}

    for block in blocks:
        title = block["title"]
        r_col = block["r_col"]
        charge_cols = block["charge_cols"]

        data[title] = {}

        for q, col in charge_cols:
            vals = []

            for row in rows[1:]:
                if r_col >= len(row):
                    continue
                if col >= len(row):
                    continue
                if not is_number(row[r_col]):
                    continue
                if not is_number(row[col]):
                    continue

                r = float(row[r_col])
                pot = float(row[col])

                if 10.0 <= r <= 14.0:
                    vals.append(pot)

            if not vals:
                raise RuntimeError(f"{title} q={q:g} 沒有 r=10–14 的資料")

            data[title][q] = sum(vals) / len(vals)

    return data


def fmt_q(q):
    return f"{q:g}"


def fmt_val(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    data = read_pot_csv(Path(args.input_csv))

    titles = ["K", "CL", "Mg"]
    if "MG" in data and "Mg" not in data:
        titles = ["K", "CL", "MG"]

    results = {}

    for title in titles:
        qs = list(data[title].keys())
        full_q = qs[-1]
        full_pot = data[title][full_q]

        rows = []
        rows.append((0.0, 0.0, 0.0))

        for q in qs:
            pot = data[title][q]
            pot_lin = (q / full_q) * full_pot
            rows.append((q, pot, pot_lin))

        results[title] = rows

    max_len = max(len(results[t]) for t in titles)

    out_rows = []

    header = []
    for i, title in enumerate(titles):
        if i > 0:
            header.append("")
        header.extend([title, "Q", "Pot", "Pot_lin"])
    out_rows.append(header)

    for idx in range(max_len):
        row = []

        for i, title in enumerate(titles):
            if i > 0:
                row.append("")

            if idx < len(results[title]):
                q, pot, pot_lin = results[title][idx]
                row.extend(["", fmt_q(q), fmt_val(pot), fmt_val(pot_lin)])
            else:
                row.extend(["", "", "", ""])

        out_rows.append(row)

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()

