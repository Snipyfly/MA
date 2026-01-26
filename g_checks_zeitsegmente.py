# checks_flanken_vs_segmente.py
from pathlib import Path
import pandas as pd

CLEANED_DIR = Path("cleaned")

SEG_FP = CLEANED_DIR / "f_zeitsegmente_match_level.csv"
OUT_FP = CLEANED_DIR / "g_check_flanken_vs_segmente_report.csv"

CROSS_PATTERNS = ["flanken_*_openplay.csv"]  # nur OpenPlay-Flanken-Tabellen

def safe_read_csv(fp: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(fp)
    except UnicodeDecodeError:
        return pd.read_csv(fp, encoding="latin-1")

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    # ------------------------------------------------------------
    # 1) Zeitsegmente laden
    # ------------------------------------------------------------
    if not SEG_FP.exists():
        raise FileNotFoundError(f"Nicht gefunden: {SEG_FP}")

    seg = safe_read_csv(SEG_FP)

    # Erwartete Spalten: match_id, team_id, crosses_openplay, minutes, ...
    # Wir prüfen robust, welche Spalten es gibt
    seg_match_col = pick_col(seg, ["match_id", "MatchId", "SourceMatchId"])
    seg_team_col  = pick_col(seg, ["team_id", "TeamId"])
    seg_cross_col = pick_col(seg, ["crosses_openplay", "crosses", "openplay_crosses"])

    missing = [x for x in [seg_match_col, seg_team_col, seg_cross_col] if x is None]
    if missing:
        raise ValueError(
            "Zeitsegmente fehlen erforderliche Spalten. "
            f"Gefunden: match={seg_match_col}, team={seg_team_col}, crosses={seg_cross_col}. "
            f"Spalten im File: {list(seg.columns)}"
        )

    # Segment-Flanken aufsummieren pro Match-Team
    seg_team_match = (
        seg.groupby([seg_match_col, seg_team_col], dropna=False)[seg_cross_col]
           .sum()
           .reset_index()
           .rename(columns={
               seg_match_col: "match_id",
               seg_team_col:  "team_id",
               seg_cross_col: "crosses_openplay_from_segments"
           })
    )

    # ------------------------------------------------------------
    # 2) OpenPlay-Flanken-Dateien laden und Team-Match zählen
    # ------------------------------------------------------------
    cross_files = []
    for pat in CROSS_PATTERNS:
        cross_files.extend(sorted(CLEANED_DIR.glob(pat)))

    if not cross_files:
        raise FileNotFoundError(f"Keine Dateien gefunden in {CLEANED_DIR} mit Pattern {CROSS_PATTERNS}")

    dfs = []
    for fp in cross_files:
        df = safe_read_csv(fp)
        df["_file"] = fp.name
        dfs.append(df)

    crosses = pd.concat(dfs, ignore_index=True)

    cross_match_col = pick_col(crosses, ["MatchId", "SourceMatchId", "match_id"])
    cross_team_col  = pick_col(crosses, ["TeamId", "team_id"])
    cross_event_col = pick_col(crosses, ["EventId", "event_id"])

    if cross_match_col is None or cross_team_col is None:
        raise ValueError(
            "In den OpenPlay-Flanken fehlen MatchId/TeamId. "
            f"Gefunden: match={cross_match_col}, team={cross_team_col}. "
            f"Spalten: {list(crosses.columns)}"
        )

    # Team-Match OpenPlay-Flanken zählen (EventId unique, falls doppelte Zeilen vorkommen)
    if cross_event_col is not None:
        cross_team_match = (
            crosses.dropna(subset=[cross_match_col, cross_team_col])
                  .assign(_eid=crosses[cross_event_col].astype(str))
                  .groupby([cross_match_col, cross_team_col])["_eid"]
                  .nunique()
                  .reset_index()
                  .rename(columns={
                      cross_match_col: "match_id",
                      cross_team_col:  "team_id",
                      "_eid": "crosses_openplay_from_files"
                  })
        )
    else:
        cross_team_match = (
            crosses.dropna(subset=[cross_match_col, cross_team_col])
                  .groupby([cross_match_col, cross_team_col])
                  .size()
                  .reset_index(name="crosses_openplay_from_files")
                  .rename(columns={
                      cross_match_col: "match_id",
                      cross_team_col:  "team_id",
                  })
        )

    # ------------------------------------------------------------
    # 3) Merge & Differenzen
    # ------------------------------------------------------------
    chk = seg_team_match.merge(
        cross_team_match,
        on=["match_id", "team_id"],
        how="outer",
        indicator=True
    )

    chk["crosses_openplay_from_segments"] = chk["crosses_openplay_from_segments"].fillna(0).astype(int)
    chk["crosses_openplay_from_files"]    = chk["crosses_openplay_from_files"].fillna(0).astype(int)
    chk["diff"] = chk["crosses_openplay_from_segments"] - chk["crosses_openplay_from_files"]

    # ------------------------------------------------------------
    # 4) Kennzahlen + Report
    # ------------------------------------------------------------
    total_pairs = len(chk)
    ok_pairs    = (chk["diff"] == 0).sum()
    bad_pairs   = (chk["diff"] != 0).sum()

    missing_in_segments = (chk["_merge"] == "right_only").sum()
    missing_in_files    = (chk["_merge"] == "left_only").sum()

    print("\n=== CHECK: OpenPlay-Flanken (Files) vs. Segment-Summen ===")
    print(f"Match-Team-Paare gesamt: {total_pairs}")
    print(f"OK (diff=0): {ok_pairs}")
    print(f"NICHT OK (diff!=0): {bad_pairs}")
    print(f"In Files, aber nicht in Segmenten: {missing_in_segments}")
    print(f"In Segmenten, aber nicht in Files: {missing_in_files}")

    if bad_pairs > 0:
        print("\nBeispiele (Top 20) mit Abweichung:")
        ex = chk.loc[chk["diff"] != 0].copy()
        ex = ex.sort_values(["diff", "match_id", "team_id"], ascending=[False, True, True]).head(20)
        print(ex[["match_id", "team_id", "crosses_openplay_from_files", "crosses_openplay_from_segments", "diff", "_merge"]].to_string(index=False))

    # Speichern
    chk.to_csv(OUT_FP, index=False)
    print(f"\nDetailreport gespeichert: {OUT_FP}")

if __name__ == "__main__":
    main()
