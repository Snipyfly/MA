import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_CROSSES_WITH_FORMATION = "cleaned/n_flanken_mit_formation_0201.csv"
DEFAULT_FORMATIONS_MATCH_TEAM = "cleaned/n_formationen_0201_match_team.csv"
DEFAULT_OUT_DIR = "cleaned"


def _is_present(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return series.notna() & (text != "") & (text.str.lower() != "nan")


def build_missing_summary(df_crosses: pd.DataFrame) -> pd.DataFrame:
    required = {"LineUp"}
    missing = required - set(df_crosses.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in Flanken-Datei: {sorted(missing)}")

    has_lineup = _is_present(df_crosses["LineUp"])
    total = len(df_crosses)
    missing_count = int((~has_lineup).sum())
    share_missing = (missing_count / total * 100.0) if total > 0 else 0.0

    rows: List[dict] = [
        {
            "Scope": "overall",
            "Season": pd.NA,
            "Competition": pd.NA,
            "CrossesTotal": total,
            "CrossesWithoutLineUp": missing_count,
            "ShareWithoutLineUpPct": round(share_missing, 4),
        }
    ]

    if {"Season", "Competition"}.issubset(df_crosses.columns):
        grouped = (
            df_crosses.assign(_has_lineup=has_lineup)
            .groupby(["Season", "Competition"], dropna=False)
            .agg(CrossesTotal=("LineUp", "size"), CrossesWithLineUp=("_has_lineup", "sum"))
            .reset_index()
        )
        grouped["CrossesWithoutLineUp"] = grouped["CrossesTotal"] - grouped["CrossesWithLineUp"]
        grouped["ShareWithoutLineUpPct"] = (
            grouped["CrossesWithoutLineUp"] / grouped["CrossesTotal"] * 100.0
        ).round(4)

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "Scope": "season_competition",
                    "Season": row["Season"],
                    "Competition": row["Competition"],
                    "CrossesTotal": int(row["CrossesTotal"]),
                    "CrossesWithoutLineUp": int(row["CrossesWithoutLineUp"]),
                    "ShareWithoutLineUpPct": row["ShareWithoutLineUpPct"],
                }
            )

    return pd.DataFrame(rows)


def build_distribution_overall(df_formations: pd.DataFrame) -> pd.DataFrame:
    required = {"LineUp"}
    missing = required - set(df_formations.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in Formations-Datei: {sorted(missing)}")

    valid = df_formations[_is_present(df_formations["LineUp"])].copy()
    total = len(valid)
    if total == 0:
        return pd.DataFrame(columns=["LineUp", "MatchTeamCount", "SharePct"])

    out = (
        valid.groupby("LineUp", dropna=False)
        .size()
        .rename("MatchTeamCount")
        .reset_index()
        .sort_values("MatchTeamCount", ascending=False)
    )
    out["SharePct"] = (out["MatchTeamCount"] / total * 100.0).round(4)
    return out


def build_distribution_by_season_comp(df_formations: pd.DataFrame) -> pd.DataFrame:
    required = {"Season", "Competition", "LineUp"}
    missing = required - set(df_formations.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in Formations-Datei: {sorted(missing)}")

    valid = df_formations[_is_present(df_formations["LineUp"])].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "Competition",
                "LineUp",
                "MatchTeamCount",
                "ShareWithinSeasonCompetitionPct",
            ]
        )

    counts = (
        valid.groupby(["Season", "Competition", "LineUp"], dropna=False)
        .size()
        .rename("MatchTeamCount")
        .reset_index()
    )
    totals = (
        valid.groupby(["Season", "Competition"], dropna=False)
        .size()
        .rename("TotalMatchTeamRows")
        .reset_index()
    )

    out = counts.merge(totals, on=["Season", "Competition"], how="left")
    out["ShareWithinSeasonCompetitionPct"] = (
        out["MatchTeamCount"] / out["TotalMatchTeamRows"] * 100.0
    ).round(4)
    out = out.drop(columns=["TotalMatchTeamRows"]).sort_values(
        ["Season", "Competition", "MatchTeamCount"], ascending=[True, True, False]
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QA fuer 02.01-Formationen: Anteil Flanken ohne Formation + "
            "Formationsverteilung (gesamt und pro Liga/Saison)."
        )
    )
    parser.add_argument(
        "--crosses-file",
        default=DEFAULT_CROSSES_WITH_FORMATION,
        help=f"Input: Flanken mit Formation (default: {DEFAULT_CROSSES_WITH_FORMATION})",
    )
    parser.add_argument(
        "--formations-file",
        default=DEFAULT_FORMATIONS_MATCH_TEAM,
        help=f"Input: Match-Team-Formationen (default: {DEFAULT_FORMATIONS_MATCH_TEAM})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output-Verzeichnis (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    crosses_path = (cwd / args.crosses_file).resolve()
    formations_path = (cwd / args.formations_file).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_crosses = pd.read_csv(crosses_path)
    df_formations = pd.read_csv(formations_path)

    summary_missing = build_missing_summary(df_crosses)
    dist_overall = build_distribution_overall(df_formations)
    dist_by_sc = build_distribution_by_season_comp(df_formations)

    out_missing = out_dir / "o_qa_flanken_ohne_formation_summary.csv"
    out_overall = out_dir / "o_qa_formation_verteilung_gesamt.csv"
    out_by_sc = out_dir / "o_qa_formation_verteilung_season_competition.csv"

    summary_missing.to_csv(out_missing, index=False)
    dist_overall.to_csv(out_overall, index=False)
    dist_by_sc.to_csv(out_by_sc, index=False)

    overall_row = summary_missing.loc[summary_missing["Scope"] == "overall"].iloc[0]
    print(f"Flanken gesamt: {int(overall_row['CrossesTotal'])}")
    print(f"Ohne Formation: {int(overall_row['CrossesWithoutLineUp'])}")
    print(f"Anteil ohne Formation (%): {overall_row['ShareWithoutLineUpPct']}")
    print(f"Gesamt-Verteilungen: {len(dist_overall)} Formationen")
    print(f"Verteilungen pro Liga/Saison: {len(dist_by_sc)} Zeilen")
    print(f"Gespeichert: {out_missing}")
    print(f"Gespeichert: {out_overall}")
    print(f"Gespeichert: {out_by_sc}")


if __name__ == "__main__":
    main()
