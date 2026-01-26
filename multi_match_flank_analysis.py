# multi_match_flank_analysis.py

import pandas as pd
from datetime import datetime
from collections import defaultdict

# Importiere die Match-Liste (BL + 2. BL, letzte 5 Seasons) + Metadaten
from season_scanner_bl_bl2 import get_match_ids_last_5_seasons

# Import aus deinem bestehenden MA-Datenimport-Modul
from ma_datenimport import (
    enrich_crosses_with_state,
    link_crosses_to_shots_for_match,
)

# ------------------------------------------------------------
# KONFIGURATION
# ------------------------------------------------------------

# Nur Spiele mit Kickoff in der Vergangenheit auswerten?
ONLY_PAST_MATCHES = True


def main():
    # 1) Alle Matches (BL + 2. BL) der letzten 5 Seasons holen
    seasons, match_ids, meta_by_mid = get_match_ids_last_5_seasons()
    print("Verwendete Seasons (letzte 5):", seasons)
    print("Anzahl gefundener Spiele (BL + 2. BL):", len(match_ids))

    # MatchIds sortieren (z.B. alphabetisch) – Reihenfolge ist später pro Season
    match_ids = sorted(match_ids)

    # 2) Matches nach Season gruppieren
    matches_by_season: dict[str, list[str]] = defaultdict(list)
    for mid in match_ids:
        meta = meta_by_mid.get(mid)
        if not meta:
            continue
        season_id = meta.get("season")
        if season_id is None:
            continue
        matches_by_season[season_id].append(mid)

    # Saisons in der Reihenfolge aus `seasons` durchgehen
    # (falls du lieber chronologisch möchtest, kannst du sort(seasons) nehmen)
    all_crosses_global = []  # flanken über alle Seasons
    all_links_global = []    # Flanke->Schuss über alle Seasons

    now = datetime.now()

    for season_id in seasons:
        mids_in_season = matches_by_season.get(season_id, [])
        if not mids_in_season:
            print(f"\n=== Season {season_id}: keine Matches gefunden ===")
            continue

        print(f"\n=== Season {season_id}: {len(mids_in_season)} Matches im Schedule ===")

        # Innerhalb der Saison: nach Kickoff sortieren (neueste zuerst)
        def kickoff_key(mid: str):
            m = meta_by_mid.get(mid, {})
            ko = m.get("kickoff")
            # None nach hinten
            return ko or datetime.min

        mids_in_season_sorted = sorted(mids_in_season, key=kickoff_key, reverse=True)

        season_crosses = []
        season_links = []

        processed_matches = 0

        for idx, mid in enumerate(mids_in_season_sorted, start=1):
            meta = meta_by_mid.get(mid, {})

            home_name = meta.get("home_name") or meta.get("home_id") or "UnbekanntHeim"
            guest_name = meta.get("guest_name") or meta.get("guest_id") or "UnbekanntGast"
            ko = meta.get("kickoff")

            if isinstance(ko, datetime):
                ko_str = ko.strftime("%Y-%m-%d %H:%M")
            else:
                ko_str = "Kickoff unbekannt"

            # Optional: nur Spiele in der Vergangenheit
            if ONLY_PAST_MATCHES and isinstance(ko, datetime) and ko > now:
                print(f"[{idx}] {mid}: {home_name} vs {guest_name} am {ko_str} – Spiel liegt in der Zukunft, übersprungen.")
                continue

            print(f"\n[{idx}] {mid}: {home_name} vs {guest_name} am {ko_str} – versuche zu verarbeiten ...")

            try:
                # a) Flanken inkl. Spielstand & rote Karten (03.05 + 03.06)
                crosses_enriched = enrich_crosses_with_state(mid)

                if crosses_enriched is None or crosses_enriched.empty:
                    print("  -> keine Flanken oder kein verwertbarer 03.06-Feed (AdvancedEvents leer/nicht vorhanden).")
                    continue

                processed_matches += 1
                print(f"  -> {len(crosses_enriched)} Flanken gefunden (Season {season_id}, Match #{processed_matches}).")
                season_crosses.append(crosses_enriched)
                all_crosses_global.append(crosses_enriched)

                # b) Flanke → Schuss-Verknüpfung (inkl. xG + Kontext)
                links_df = link_crosses_to_shots_for_match(mid, time_window_seconds=8.0)
                if links_df is not None and not links_df.empty:
                    print(f"  -> {len(links_df)} Flanke→Schuss-Ketten gefunden.")
                    season_links.append(links_df)
                    all_links_global.append(links_df)
                else:
                    print("  -> keine Flanke→Schuss-Ketten im Zeitfenster (oder keine Schüsse).")

            except Exception as e:
                print(f"  !! Fehler bei Match {mid}: {e}")
                continue

        # --- Season-Ergebnisse zusammenführen + speichern ---
        if season_crosses:
            df_crosses_season = pd.concat(season_crosses, ignore_index=True)
            print(f"\n=== Season {season_id}: Gesamt Flanken ===")
            print("Gesamtanzahl Flanken:", len(df_crosses_season))

            # kleine Übersicht
            cols_cross_overview = [
                "MatchId",
                "EventId", "TeamId", "GameTime",
                "X", "Y", "X_rec", "Y_rec",
                "NumDefInBox", "NumAttInBox",
                "ScoreHome", "ScoreGuest", "MatchStateForTeam",
                "AnyRedBeforeCross", "RedForOwnTeam", "RedForOppTeam",
            ]
            cols_cross_overview = [c for c in cols_cross_overview if c in df_crosses_season.columns]

            print(
                df_crosses_season[cols_cross_overview].head(20).to_string(index=False)
            )

            # Nur CSV speichern (kein Parquet)
            fname_cross_csv = f"flanken_{season_id}.csv"
            df_crosses_season.to_csv(fname_cross_csv, index=False)
            print(f"Season {season_id}: Flanken-Daten gespeichert als '{fname_cross_csv}'.")
        else:
            print(f"\nSeason {season_id}: Keine Flanken-Daten gesammelt (kein Match lieferte AdvancedEvents/Flanken).")

        if season_links:
            df_links_season = pd.concat(season_links, ignore_index=True)
            print(f"\n=== Season {season_id}: Flanke → Schuss ===")
            print("Gesamtanzahl Flanke→Schuss-Ketten:", len(df_links_season))

            cols_links_overview = [
                "MatchId",

                # Flanke (aus Plays/enriched)
                "EventId",         # Flanken-Event
                "TeamId",
                "PlayerId",
                "GameTime",
                "X", "Y",
                "X_rec", "Y_rec",
                "NumDefInBox", "NumAttInBox",

                # Kontext
                "ScoreHome", "ScoreGuest",
                "MatchStateForTeam",
                "AnyRedBeforeCross", "RedForOwnTeam", "RedForOppTeam",

                # Schuss
                "ShotEventId",
                "ShotPlayerId",
                "ShotGameTime",
                "ShotxG",
                "ShotResult",
            ]
            cols_links_overview = [c for c in cols_links_overview if c in df_links_season.columns]

            print(
                df_links_season[cols_links_overview].head(20).to_string(index=False)
            )

            fname_links_csv = f"flanke_shot_links_{season_id}.csv"
            df_links_season.to_csv(fname_links_csv, index=False)
            print(f"Season {season_id}: Flanke→Schuss-Daten gespeichert als '{fname_links_csv}'.")
        else:
            print(f"\nSeason {season_id}: Keine Flanke→Schuss-Ketten gesammelt.")

if __name__ == "__main__":
    main()
