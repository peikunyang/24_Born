#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

COULOMB = 332.06371


def is_number(x):
    try:
        float(str(x).strip())
        return True
    except:
        return False


def fmt_q(x):
    return f"{x:g}"


def fmt_val(x):
    return f"{x:.3f}".rstrip("0").rstrip(".")


def read_delta_pot_csv(path, r_min, r_max):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

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
                if is_number(h):
                    q_cols.append((float(h), j))
                j += 1

            blocks.append({
                "title": title,
                "r_col": r_col,
                "q_cols": q_cols,
            })

            i = j + 1
        else:
            i += 1

    data = {}

    for block in blocks:
        title = block["title"]
        r_col = block["r_col"]
        q_cols = block["q_cols"]

        data[title] = {}

        for q, col in q_cols:
            vals = []

            for row in rows[1:]:
                if r_col >= len(row) or col >= len(row):
                    continue
                if not is_number(row[r_col]) or not is_number(row[col]):
                    continue

                r = float(row[r_col])
                pot = float(row[col])

                if r_min <= r <= r_max:
                    vals.append(pot)

            if not vals:
                raise RuntimeError(f"{title} Q={q:g} 沒有 r={r_min}-{r_max} Å 的資料")

            data[title][q] = sum(vals) / len(vals)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--eps", type=float, default=80.0)
    parser.add_argument("--r_min", type=float, default=10.0)
    parser.add_argument("--r_max", type=float, default=14.0)
    args = parser.parse_args()

    data = read_delta_pot_csv(Path(args.input_csv), args.r_min, args.r_max)

    titles = ["K", "CL", "Mg"]
    if "MG" in data and "Mg" not in data:
        titles = ["K", "CL", "MG"]

    factor = 1.0 - 1.0 / args.eps

    results = {}

    for title in titles:
        results[title] = []

        for q in sorted(data[title].keys()):
            pot = data[title][q]

            if abs(pot) < 1e-12:
                bor_rad = ""
            else:
                bor_rad = -COULOMB * factor * q / pot

            results[title].append((q, bor_rad))

    max_len = max(len(results[t]) for t in titles)

    out_rows = []

    header = []
    for i, title in enumerate(titles):
        if i > 0:
            header.append("")
        header.extend([title, "Q", "Bor_rad"])
    out_rows.append(header)

    for row_i in range(max_len):
        row = []

        for i, title in enumerate(titles):
            if i > 0:
                row.append("")

            if row_i < len(results[title]):
                q, bor_rad = results[title][row_i]
                row.extend([
                    "",
                    fmt_q(q),
                    fmt_val(bor_rad) if bor_rad != "" else "",
                ])
            else:
                row.extend(["", "", ""])

        out_rows.append(row)

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()

