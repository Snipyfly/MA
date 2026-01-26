# daten_uebersicht.py
import re
from pathlib import Path
import pandas as pd

# ============================================================
# KONFIG
# ============================================================
INPUT_DIR = Path("cleaned")  # hier liegen deine *_openplay.csv
OUTPUT_DIR = INPUT_DIR

PATTERNS = [
    "flanken*_openplay.csv",
    "flanke_shot_links*_openplay.csv",
]

OUTPUT_CSV = OUTPUT_DIR / "grunddaten_uebersicht.csv"
OUTPUT_CSV_FILES = OUTPUT_DIR / "grunddaten_uebersicht_dateien.csv"
OUTPUT_CSV_MATCHES = OUTPUT_DIR / "grunddaten_uebersicht_matches.csv"


# ============================================================
# HELFER
# ============================================================

def safe_read_csv(fp: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(fp)
    except UnicodeDecodeError:
        return pd.read_csv(fp, encoding="latin-1")


def parse_meta_from_filename(name: str) -> dict:
    """
    Extrahiert Liga/Season aus Dateinamen wie:
      flanken_1_BL_DFL-SEA-0001K7_openplay.csv
      flanke_shot_links_2_BL_DFL-SEA-0001K8_openplay.csv
    """
    meta = {"league": None, "season": None, "table_type": None}

    # Tabellenart
    if name.startswith("flanke_shot_links"):
        meta["table_type"] = "cross_shot_links"
    elif name.startswith("flanken"):
        meta["table_type"] = "crosses"
    else:
        meta["table_type"] = "unknown"

    # Liga (1_BL / 2_BL)
    m_league = re.search(r"_(1_BL|2_BL)_", name)
    if m_league:
        meta["league"] = "Bundesliga" if m_league.group(1) == "1_BL" else "2. Bundesliga"

    # SeasonId
    m_season = re.search(r"(DFL-SEA-[A-Z0-9\-]+)", name)
    if m_season:
        meta["season"] = m_season.group(1)

    return meta


def pick_match_id_column(df: pd.DataFrame) -> str | None:
    """
    Einheitliche MatchId-Erkennung.
    Priorität:
      1) MatchId (aus Feed/Export)
      2) SourceMatchId (von dir gesetzt; in Shot-Link Tabellen typisch)
      3) Cross_MatchId / CrossMatchId (falls du mal prefixed hattest)
    """
    candidates = ["MatchId", "SourceMatchId", "Cross_MatchId", "CrossMatchId"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stellt sicher, dass es eine Spalte _match_id gibt (string),
    egal ob MatchId oder SourceMatchId im File steht.
    """
    df = df.copy()
    col = pick_match_id_column(df)
    if col is None:
        df["_match_id"] = pd.NA
        return df
    df["_match_id"] = df[col].astype("string")
    return df


def normalize_event_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stellt sicher, dass EventId als string vorliegt (für Join / isin).
    """
    df = df.copy()
    if "EventId" in df.columns:
        df["_event_id"] = df["EventId"].astype("string")
    else:
        df["_event_id"] = pd.NA
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    # 1) Dateien finden
    files: list[Path] = []
    for pat in PATTERNS:
        files.extend(sorted(INPUT_DIR.glob(pat)))

    if not files:
        raise FileNotFoundError(f"Keine Dateien gefunden in {INPUT_DIR} mit Patterns: {PATTERNS}")

    print(f"Gefundene Dateien: {len(files)}")
    print(f"Input: {INPUT_DIR}")

    per_file_rows = []
    all_crosses = []  # nur flanken-tabellen
    all_links = []    # nur shot-links-tabellen

    # 2) pro Datei lesen + Kennzahlen
    for fp in files:
        meta = parse_meta_from_filename(fp.name)
        df = safe_read_csv(fp)

        # Normalisierung (match id + event id)
        df = normalize_match_id(df)
        df = normalize_event_id(df)

        uniq_matches = df["_match_id"].nunique(dropna=True) if "_match_id" in df.columns else pd.NA

        per_file_rows.append({
            "file": fp.name,
            "table_type": meta["table_type"],
            "league_from_filename": meta["league"],
            "season_from_filename": meta["season"],
            "rows": len(df),
            "match_id_col_used": pick_match_id_column(df),
            "unique_matches": uniq_matches,
        })

        # Metadaten aus Dateiname an DataFrame hängen
        df["_file"] = fp.name
        df["_league_file"] = meta["league"]
        df["_season_file"] = meta["season"]

        # Sammeln
        if meta["table_type"] == "crosses":
            all_crosses.append(df)
        elif meta["table_type"] == "cross_shot_links":
            all_links.append(df)

    df_files = pd.DataFrame(per_file_rows).sort_values(
        ["table_type", "season_from_filename", "league_from_filename", "file"]
    )
    df_files.to_csv(OUTPUT_CSV_FILES, index=False)
    print(f"Datei-Übersicht gespeichert: {OUTPUT_CSV_FILES}")

    # 3) Gesamt zusammenführen
    df_crosses_all = pd.concat(all_crosses, ignore_index=True) if all_crosses else pd.DataFrame()
    df_links_all = pd.concat(all_links, ignore_index=True) if all_links else pd.DataFrame()

    # 4) Aggregation Crosses nach Season/Liga (aus Dateinamen)
    overview_rows = []

    if not df_crosses_all.empty:
        cross_grp = df_crosses_all.groupby(["_season_file", "_league_file"], dropna=False)

        for (season, league), g in cross_grp:
            crosses = len(g)
            matches = g["_match_id"].nunique(dropna=True)

            # Share: nur Links derselben Season/Liga (wichtig!)
            share = pd.NA
            linked_crosses = pd.NA
            if not df_links_all.empty and g["_event_id"].notna().any():
                links_sub = df_links_all[
                    (df_links_all["_season_file"] == season) &
                    (df_links_all["_league_file"] == league)
                ]
                if not links_sub.empty and links_sub["_event_id"].notna().any():
                    link_event_ids = set(links_sub["_event_id"].dropna())
                    n_linked = g["_event_id"].dropna().isin(link_event_ids).sum()
                    linked_crosses = int(n_linked)
                    share = (n_linked / crosses) if crosses > 0 else 0.0
                else:
                    linked_crosses = 0
                    share = 0.0

            overview_rows.append({
                "season": season,
                "league": league,
                "matches": matches,
                "crosses_openplay": crosses,
                "crosses_with_shot_link": linked_crosses,
                "share_crosses_with_shot_link": share,
            })

    df_overview = pd.DataFrame(overview_rows).sort_values(["season", "league"])
    df_overview.to_csv(OUTPUT_CSV, index=False)
    print(f"Grunddaten-Übersicht gespeichert: {OUTPUT_CSV}")

    # 5) Match-level Übersicht (Crosses + optional Links)
    match_rows = []
    if not df_crosses_all.empty:
        # Links pro Match (für Join)
        links_per_match = None
        if not df_links_all.empty:
            links_per_match = (
                df_links_all.groupby(["_season_file", "_league_file", "_match_id"], dropna=False)
                .size()
                .reset_index(name="shot_links")
            )

        crosses_per_match = (
            df_crosses_all.groupby(["_season_file", "_league_file", "_match_id"], dropna=False)
            .size()
            .reset_index(name="crosses_openplay")
            .rename(columns={"_season_file": "season", "_league_file": "league", "_match_id": "match_id"})
        )

        if links_per_match is not None and not links_per_match.empty:
            links_per_match = links_per_match.rename(
                columns={"_season_file": "season", "_league_file": "league", "_match_id": "match_id"}
            )
            crosses_per_match = crosses_per_match.merge(
                links_per_match,
                on=["season", "league", "match_id"],
                how="left"
            )
            crosses_per_match["shot_links"] = crosses_per_match["shot_links"].fillna(0).astype(int)
        else:
            crosses_per_match["shot_links"] = 0

        # pro Match noch Quote
        crosses_per_match["share_links_per_cross"] = crosses_per_match.apply(
            lambda r: (r["shot_links"] / r["crosses_openplay"]) if r["crosses_openplay"] > 0 else 0.0,
            axis=1
        )

        df_matches = crosses_per_match.sort_values(["season", "league", "match_id"])
        df_matches.to_csv(OUTPUT_CSV_MATCHES, index=False)
        print(f"Match-Übersicht gespeichert: {OUTPUT_CSV_MATCHES}")
    else:
        df_matches = pd.DataFrame()
        df_matches.to_csv(OUTPUT_CSV_MATCHES, index=False)
        print(f"Match-Übersicht gespeichert (leer): {OUTPUT_CSV_MATCHES}")

    # 6) Konsolen-Output (kompakt)
    print("\n=== KOMPAKTÜBERSICHT (Open-Play-Flanken) ===")
    if df_overview.empty:
        print("Keine Flanken-Daten (crosses) gefunden.")
    else:
        for season in df_overview["season"].dropna().unique():
            sub = df_overview[df_overview["season"] == season]

            total_matches = int(sub["matches"].sum())  # BL und 2.BL sind disjunkt -> Summe ok
            total_crosses = int(sub["crosses_openplay"].sum())

            print(f"\n{season}: Matches={total_matches} | Flanken(OpenPlay)={total_crosses}")
            for _, r in sub.iterrows():
                share = r["share_crosses_with_shot_link"]
                share_str = f"{share:.3f}" if pd.notna(share) else "—"
                print(
                    f"  - {r['league']}: Matches={int(r['matches'])} | "
                    f"Flanken={int(r['crosses_openplay'])} | Anteil->Schuss={share_str}"
                )

    # 7) Extra: Gesamtzahlen Links (Match-Zahl jetzt korrekt!)
    print("\n=== Shot-Link Gesamt ===")
    if df_links_all.empty:
        print("Keine Shot-Link-Daten gefunden.")
    else:
        n_links = int(len(df_links_all))
        uniq_matches_links = int(df_links_all["_match_id"].nunique(dropna=True))
        print(f"Flanke→Schuss-Zeilen: {n_links}")
        print(f"Matches mit mindestens 1 Link: {uniq_matches_links}")


if __name__ == "__main__":
    main()
