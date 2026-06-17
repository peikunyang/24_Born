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


def fmt_q(x):
    return f"{x:g}"


def fmt_val(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)

    with input_csv.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header = rows[0]

    blocks = []
    i = 0

    while i < len(header):
        title = header[i].strip()

        if title in {"K", "CL", "Mg", "MG"}:
            q_col = i + 1
            pot_col = i + 2
            pot_lin_col = i + 3

            blocks.append({
                "title": title,
                "q_col": q_col,
                "pot_col": pot_col,
                "pot_lin_col": pot_lin_col,
            })

            i += 5
        else:
            i += 1

    results = {}

    for block in blocks:
        title = block["title"]
        q_col = block["q_col"]
        pot_col = block["pot_col"]
        pot_lin_col = block["pot_lin_col"]

        results[title] = []

        for row in rows[1:]:
            if pot_lin_col >= len(row):
                continue

            q_txt = row[q_col].strip()
            pot_txt = row[pot_col].strip()
            pot_lin_txt = row[pot_lin_col].strip()

            if not (is_number(q_txt) and is_number(pot_txt) and is_number(pot_lin_txt)):
                continue

            q = float(q_txt)
            pot = float(pot_txt)
            pot_lin = float(pot_lin_txt)

            pot_err = pot_lin - pot

            results[title].append((q, pot_err))

    titles = [b["title"] for b in blocks]
    max_len = max(len(results[t]) for t in titles)

    out_rows = []

    header_out = []
    for idx, title in enumerate(titles):
        if idx > 0:
            header_out.append("")
        header_out.extend([title, "Q", "Pot_Err"])
    out_rows.append(header_out)

    for row_i in range(max_len):
        out_row = []

        for idx, title in enumerate(titles):
            if idx > 0:
                out_row.append("")

            if row_i < len(results[title]):
                q, pot_err = results[title][row_i]
                out_row.extend(["", fmt_q(q), fmt_val(pot_err)])
            else:
                out_row.extend(["", "", ""])

        out_rows.append(out_row)

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()

