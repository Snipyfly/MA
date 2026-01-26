# batch_clean_cross_csvs.py
import os
import glob
import pandas as pd


STANDARD_COLS = ["IsCorner", "IsFreeKick", "IsGoalKick", "IsThrowIn", "IsKickOff"]


def filter_open_play_crosses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtert ausschließlich Flanken aus dem Spiel (keine Standards).
    Erwartet mindestens IsCross.
    """
    df = df.copy()

    if "IsCross" not in df.columns:
        raise ValueError("Spalte 'IsCross' fehlt – kann nicht als Flanken-CSV interpretiert werden.")

    # 1) Nur Flanken
    df = df[df["IsCross"] == True]

    # 2) Standards ausschließen (alles, was explizit True ist)
    for col in STANDARD_COLS:
        if col in df.columns:
            df = df[df[col] != True]

    return df.reset_index(drop=True)


def clean_csv_file(in_path: str, out_dir: str) -> tuple[str, int, int]:
    """
    Lädt CSV, bereinigt, speichert bereinigte Version.
    Gibt (out_path, n_before, n_after) zurück.
    """
    df = pd.read_csv(in_path)

    n_before = len(df)
    df_clean = filter_open_play_crosses(df)
    n_after = len(df_clean)

    base = os.path.basename(in_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}_openplay{ext}")

    df_clean.to_csv(out_path, index=False)
    return out_path, n_before, n_after


def main():
    # Ordner, in dem deine CSVs liegen (Standard: aktuelles Working Directory)
    base_dir = os.getcwd()

    # Zielordner für bereinigte Dateien
    out_dir = os.path.join(base_dir, "cleaned")
    os.makedirs(out_dir, exist_ok=True)

    # Welche CSVs sollen bereinigt werden?
    # - flanken_*.csv (alle Flanken)
    # - flanke_shot_links_*.csv (Flanken mit anschließendem Schuss)
    patterns = [
        os.path.join(base_dir, "flanken_*.csv"),
        os.path.join(base_dir, "flanke_shot_links_*.csv"),
    ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    files = sorted(set(files))

    if not files:
        print("Keine passenden CSV-Dateien gefunden. Erwartet z. B. 'flanken_*.csv' oder 'flanke_shot_links_*.csv'.")
        return

    print(f"Gefundene Dateien: {len(files)}")
    print(f"Ausgabe nach: {out_dir}")
    print("-" * 60)

    summary_rows = []

    for f in files:
        try:
            out_path, n_before, n_after = clean_csv_file(f, out_dir)
            removed = n_before - n_after
            print(f"{os.path.basename(f)} -> {os.path.basename(out_path)} | {n_before} -> {n_after} (entfernt: {removed})")
            summary_rows.append({
                "file_in": os.path.basename(f),
                "rows_before": n_before,
                "rows_after": n_after,
                "removed": removed,
                "file_out": os.path.basename(out_path),
            })
        except Exception as e:
            print(f"FEHLER bei {os.path.basename(f)}: {e}")

    # Summary als CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "d_cleaning_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("-" * 60)
    print(f"Summary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
