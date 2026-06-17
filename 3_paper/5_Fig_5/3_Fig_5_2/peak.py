#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

ION_SIGN_SEQ = {
    "K":  [-1,  1, -1,  1],
    "CL": [ 1, -1,  1, -1],
    "MG": [-1,  1, -1,  1],
    "Mg": [-1,  1, -1,  1],
}

ROWS = ["P1st", "V1st", "P2nd", "V2nd"]


def is_number(x):
    try:
        float(str(x).strip())
        return True
    except:
        return False


def fmt(x, digits=3):
    if x is None:
        return ""
    s = f"{x:.{digits}f}"
    return s.rstrip("0").rstrip(".")


def read_fig5_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    blocks = []

    i = 0
    while i < len(header):
        title = header[i].strip()

        if title in {"K", "CL", "MG", "Mg"}:
            q_cols = []
            j = i + 1

            while j < len(header):
                h = header[j].strip()

                if h in {"K", "CL", "MG", "Mg"}:
                    break
                if h == "":
                    break
                if is_number(h):
                    q_cols.append((h, j))
                j += 1

            blocks.append((title, q_cols))
            i = j
        else:
            i += 1

    data = {}

    for title, q_cols in blocks:
        data[title] = {}

        for q_label, col in q_cols:
            curve = []

            for row in rows[1:]:
                if len(row) <= col:
                    continue
                if not is_number(row[0]):
                    continue
                if not is_number(row[col]):
                    continue

                r = float(row[0])
                y = float(row[col])
                curve.append((r, y))

            data[title][q_label] = curve

    return data


def get_dr(curve):
    rs = [r for r, y in curve]
    diffs = [round(rs[i + 1] - rs[i], 10) for i in range(len(rs) - 1)]
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        raise RuntimeError("無法判斷 dr")
    return sorted(diffs)[len(diffs) // 2]


def sign_of(y, zero_tol):
    if y > zero_tol:
        return 1
    if y < -zero_tol:
        return -1
    return 0


def split_segments(curve, zero_tol):
    segments = []
    current_sign = 0
    current = []

    for r, y in curve:
        s = sign_of(y, zero_tol)

        if s == 0:
            if current:
                segments.append((current_sign, current))
                current = []
                current_sign = 0
            continue

        if not current:
            current = [(r, y)]
            current_sign = s
        elif s == current_sign:
            current.append((r, y))
        else:
            segments.append((current_sign, current))
            current = [(r, y)]
            current_sign = s

    if current:
        segments.append((current_sign, current))

    return segments


def calc_one_curve(curve, sign_seq, zero_tol, min_abs_charge):
    dr = get_dr(curve)
    segments = split_segments(curve, zero_tol)

    results = []
    pos = 0

    for wanted_sign in sign_seq:
        found = None

        while pos < len(segments):
            seg_sign, seg = segments[pos]
            pos += 1

            acc_charge = sum(y for r, y in seg) * dr

            if seg_sign != wanted_sign:
                continue
            if abs(acc_charge) < min_abs_charge:
                continue

            found = (seg_sign, seg, acc_charge)
            break

        if found is None:
            results.append((None, None))
            continue

        seg_sign, seg, acc_charge = found

        if seg_sign > 0:
            peak_r, peak_y = max(seg, key=lambda x: x[1])
        else:
            peak_r, peak_y = min(seg, key=lambda x: x[1])

        results.append((peak_r, acc_charge))

    return results


def fold_value(a1, a2, mode):
    if a1 is None or a2 is None:
        return None
    if abs(a1) < 1e-12 or abs(a2) < 1e-12:
        return None

    if mode == "first_over_second":
        return abs(a1) / abs(a2)
    if mode == "second_over_first":
        return abs(a2) / abs(a1)

    raise RuntimeError("fold_mode 錯誤")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--zero_tol", type=float, default=0.0)
    parser.add_argument("--min_abs_charge", type=float, default=0.0)
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument(
        "--fold_mode",
        choices=["first_over_second", "second_over_first"],
        default="first_over_second",
    )
    args = parser.parse_args()

    data = read_fig5_csv(Path(args.input_csv))

    out_rows = []

    for title in ["K", "CL", "MG"]:
        if title not in data:
            continue

        q_labels = list(data[title].keys())
        if len(q_labels) != 2:
            raise RuntimeError(f"{title} 需要剛好兩個 q 欄位，目前是 {q_labels}")

        q1, q2 = q_labels[0], q_labels[1]

        res1 = calc_one_curve(
            data[title][q1],
            ION_SIGN_SEQ[title],
            args.zero_tol,
            args.min_abs_charge,
        )
        res2 = calc_one_curve(
            data[title][q2],
            ION_SIGN_SEQ[title],
            args.zero_tol,
            args.min_abs_charge,
        )

        out_rows.append([title])
        out_rows.append(["", "rho_P", "rho_P", "rho_A", "rho_A", "fold"])
        out_rows.append(["q", q1, q2, q1, q2, "fold"])

        for i, row_name in enumerate(ROWS):
            p1, a1 = res1[i]
            p2, a2 = res2[i]
            fold = fold_value(a1, a2, args.fold_mode)

            out_rows.append([
                row_name,
                fmt(p1, args.digits),
                fmt(p2, args.digits),
                fmt(a1, args.digits),
                fmt(a2, args.digits),
                fmt(fold, args.digits),
            ])

        out_rows.append([])

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()

