# i_diagnose_event_assignment.py
import re
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# KONFIG
# ============================================================
CLEANED_DIR = Path("cleaned")

CROSS_PATTERNS = [
    "flanken_1_BL_*_openplay.csv",
    "flanken_2_BL_*_openplay.csv",
]

SEGMENTS_FP = CLEANED_DIR / "f_zeitsegmente_match_level.csv"

OUT_EVENT = CLEANED_DIR / "i_diagnose_eventid.csv"
OUT_SUMMARY = CLEANED_DIR / "i_diagnose_summary.csv"


# ============================================================
# HELFER: robustes Laden
# ============================================================
def safe_read_csv(fp: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(fp)
    except UnicodeDecodeError:
        return pd.read_csv(fp, encoding="latin-1")


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# ZEIT-PARSING
# ============================================================
def parse_dfl_gametime_to_seconds(x) -> float:
    """
    Erwartet i.d.R. DFL-Format: MMM:SS:CC (Minuten : Sekunden : Hundertstel)
    Beispiele:
      '001:05:56' -> 65.56 Sekunden (Minute 1, Sek 5, 56 Hundertstel)
      '090:12:03' -> 5412.03 Sekunden
    Unterstützt zusätzlich:
      - 'MM:SS' (ohne Hundertstel)
      - numerisch (falls schon Sekunden)
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()

    # Wenn es rein numerisch ist (selten), direkt interpretieren
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)

    parts = s.split(":")
    try:
        if len(parts) == 3:
            m = int(parts[0])
            sec = int(parts[1])
            cc = int(parts[2])
            return m * 60 + sec + cc / 100.0
        elif len(parts) == 2:
            m = int(parts[0])
            sec = int(parts[1])
            return m * 60 + sec
        else:
            return np.nan
    except Exception:
        return np.nan


def seconds_to_minute_of_play(sec: float) -> float:
    """
    Minute-of-play (1-basiert) aus Sekunden:
      0.00s .. 59.99s -> Minute 1
      60.00s .. 119.99s -> Minute 2
    """
    if pd.isna(sec):
        return np.nan
    if sec < 0:
        return np.nan
    return float(int(np.floor(sec / 60.0)) + 1)


def is_boundary_case(sec: float, threshold_seconds: float = 1.0) -> bool:
    """
    Grenzfälle: sehr nah an Minutenwechseln.
    z.B. xx:00.00 bis xx:00.99 oder xx:59.00 bis xx:59.99 (bei threshold=1s).
    """
    if pd.isna(sec):
        return False
    frac = sec % 60.0
    return (frac < threshold_seconds) or (frac > (60.0 - threshold_seconds))


def stoppage_flag(minute_of_play: float, in_game_section: str | None) -> bool:
    """
    Nachspielzeit-Flag:
    - firstHalf: alles >45
    - secondHalf: alles >90
    Falls InGameSection fehlt: nur >90 als konservatives Kriterium.
    """
    if pd.isna(minute_of_play):
        return False
    if in_game_section is None or pd.isna(in_game_section):
        return minute_of_play > 90
    s = str(in_game_section).lower()
    if "first" in s:
        return minute_of_play > 45
    if "second" in s:
        return minute_of_play > 90
    return minute_of_play > 90


# ============================================================
# SEGMENT-FILE: Spalten finden + normalisieren
# ============================================================
def load_segments() -> pd.DataFrame:
    if not SEGMENTS_FP.exists():
        raise FileNotFoundError(
            f"Segment-Datei nicht gefunden: {SEGMENTS_FP}\n"
            f"Erwartet wird deine Match-Level Segmentdatei aus zeitsegmente_und_flankenraten.py."
        )

    seg = safe_read_csv(SEGMENTS_FP)

    # zwingende Schlüsselfelder
    match_col = pick_first_existing(seg, ["match_id", "MatchId"])
    team_col = pick_first_existing(seg, ["team_id", "TeamId"])
    if match_col is None or team_col is None:
        raise ValueError("Segmentdatei braucht match_id und team_id (oder MatchId/TeamId).")

    # start/end Minute-of-play Felder (dein Output hatte z.B. start_mop/end_mop o.Ä.)
    start_col = pick_first_existing(seg, ["start_mop", "start_minute_of_play", "start_min", "start_minute"])
    end_col = pick_first_existing(seg, ["end_mop", "end_minute_of_play", "end_min", "end_minute"])

    if start_col is None or end_col is None:
        raise ValueError(
            "Segmentdatei braucht Start/Ende-Spalten, z.B. start_mop/end_mop "
            "oder start_minute_of_play/end_minute_of_play."
        )

    # Normalisieren
    seg = seg.rename(
        columns={
            match_col: "match_id",
            team_col: "team_id",
            start_col: "seg_start_mop",
            end_col: "seg_end_mop",
        }
    )

    # Sicherstellen numeric
    seg["seg_start_mop"] = pd.to_numeric(seg["seg_start_mop"], errors="coerce")
    seg["seg_end_mop"] = pd.to_numeric(seg["seg_end_mop"], errors="coerce")

    # Minimale Sanity:
    seg = seg.dropna(subset=["match_id", "team_id", "seg_start_mop", "seg_end_mop"])

    return seg


# ============================================================
# FLANKEN-EVENTS LADEN + NORMALISIEREN
# ============================================================
def load_cross_events() -> pd.DataFrame:
    fps = []
    for pat in CROSS_PATTERNS:
        fps.extend(sorted(CLEANED_DIR.glob(pat)))

    if not fps:
        raise FileNotFoundError(f"Keine Flanken-OpenPlay-Dateien gefunden in {CLEANED_DIR} mit {CROSS_PATTERNS}")

    frames = []
    for fp in fps:
        df = safe_read_csv(fp)
        df["_src_file"] = fp.name
        frames.append(df)

    c = pd.concat(frames, ignore_index=True)

    # Kernspalten ermitteln
    match_col = pick_first_existing(c, ["MatchId", "match_id", "SourceMatchId"])
    team_col = pick_first_existing(c, ["TeamId", "team_id", "CrossTeamId"])
    event_col = pick_first_existing(c, ["EventId", "event_id", "CrossEventId"])

    # Zeitspalten: bevorzugt GameTime (wenn vorhanden), sonst EventTime/CrossTime
    gametime_col = pick_first_existing(c, ["GameTime", "game_time"])
    ingame_section_col = pick_first_existing(c, ["InGameSection", "GameSection", "game_section"])

    if match_col is None or team_col is None or event_col is None:
        raise ValueError("Flanken-Dateien brauchen MatchId/TeamId/EventId (oder äquivalente Spalten).")

    c = c.rename(
        columns={
            match_col: "match_id",
            team_col: "team_id",
            event_col: "event_id",
        }
    )

    # Zeit parsen
    if gametime_col is not None:
        c["time_source"] = "GameTime"
        c["gametime_raw"] = c[gametime_col]
        c["gametime_sec"] = c[gametime_col].map(parse_dfl_gametime_to_seconds)
    else:
        # fallback: wir können hier auch EventTime/CrossTime parsen, aber das ist feed-abhängig
        c["time_source"] = "NONE"
        c["gametime_raw"] = pd.NA
        c["gametime_sec"] = np.nan

    # InGameSection
    if ingame_section_col is not None:
        c["in_game_section"] = c[ingame_section_col]
    else:
        c["in_game_section"] = pd.NA

    # Minute-of-play
    c["minute_of_play"] = c["gametime_sec"].map(seconds_to_minute_of_play)

    # Hilfsflags
    c["time_na"] = c["gametime_sec"].isna()
    c["boundary_case"] = c["gametime_sec"].map(is_boundary_case)
    c["stoppage_time"] = [
        stoppage_flag(m, s) for m, s in zip(c["minute_of_play"], c["in_game_section"])
    ]

    return c


# ============================================================
# ASSIGNMENT-DIAGNOSE: Event -> Segmente
# ============================================================
def diagnose_assignment(events: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """
    Für jedes Event prüfen:
      - liegt minute_of_play in genau 1 Segment? (ok)
      - in 0 Segmenten? (not_assignable)
      - in >1 Segmenten? (ambiguous)
    Zusätzlich:
      - outside_segment_range: minute_of_play < min_start oder > max_end des Teams im Match
    """
    # Segment-range pro Match-Team
    rng = (
        seg.groupby(["match_id", "team_id"], as_index=False)
        .agg(seg_min_start=("seg_start_mop", "min"), seg_max_end=("seg_end_mop", "max"))
    )

    ev = events.merge(rng, on=["match_id", "team_id"], how="left")

    # outside range
    ev["outside_segment_range"] = False
    mask_has_rng = ev["seg_min_start"].notna() & ev["seg_max_end"].notna() & ev["minute_of_play"].notna()
    ev.loc[mask_has_rng, "outside_segment_range"] = (
        (ev.loc[mask_has_rng, "minute_of_play"] < ev.loc[mask_has_rng, "seg_min_start"])
        | (ev.loc[mask_has_rng, "minute_of_play"] > ev.loc[mask_has_rng, "seg_max_end"])
    )

    # Zählung Segment-Treffer (naiv aber robust; bei ~35k Events ok)
    # Wir iterieren pro Match-Team, um nicht riesige Cross-Joins zu bauen.
    hit_counts = []
    seg_group = {k: g for k, g in seg.groupby(["match_id", "team_id"])}

    for idx, row in ev[["match_id", "team_id", "minute_of_play"]].iterrows():
        mid = row["match_id"]
        tid = row["team_id"]
        mop = row["minute_of_play"]

        if pd.isna(mop) or (mid, tid) not in seg_group:
            hit_counts.append(np.nan)
            continue

        g = seg_group[(mid, tid)]
        hits = ((g["seg_start_mop"] <= mop) & (mop <= g["seg_end_mop"])).sum()
        hit_counts.append(int(hits))

    ev["n_matching_segments"] = hit_counts

    ev["not_assignable_despite_time"] = ev["minute_of_play"].notna() & (ev["n_matching_segments"] == 0)
    ev["ambiguous_multi_segment"] = ev["minute_of_play"].notna() & (ev["n_matching_segments"] > 1)

    # Sonstige Probleme (als Sammelbecken)
    ev["sonstige_probleme"] = False
    # Beispielregel: hat zwar Segment-Range, aber n_matching_segments ist NaN
    ev.loc[ev["minute_of_play"].notna() & ev["seg_min_start"].notna() & ev["n_matching_segments"].isna(), "sonstige_probleme"] = True

    ev["any_issue"] = (
        ev["time_na"]
        | ev["stoppage_time"]
        | ev["boundary_case"]
        | ev["outside_segment_range"]
        | ev["not_assignable_despite_time"]
        | ev["ambiguous_multi_segment"]
        | ev["sonstige_probleme"]
    )

    return ev


# ============================================================
# MAIN
# ============================================================
def main():
    seg = load_segments()
    ev = load_cross_events()

    # Diagnose
    ev2 = diagnose_assignment(ev, seg)

    # Summary
    total = len(ev2)

    summary = [
        ("total_openplay_crosses_events", total),
        ("time_na", int(ev2["time_na"].sum())),
        ("stoppage_time", int(ev2["stoppage_time"].sum())),
        ("grenzfaelle_boundary_case", int(ev2["boundary_case"].sum())),
        ("outside_segment_range", int(ev2["outside_segment_range"].sum())),
        ("not_assignable_despite_time", int(ev2["not_assignable_despite_time"].sum())),
        ("ambiguous_multi_segment", int(ev2["ambiguous_multi_segment"].sum())),
        ("sonstige_probleme", int(ev2["sonstige_probleme"].sum())),
        ("any_issue", int(ev2["any_issue"].sum())),
    ]

    df_sum = pd.DataFrame(summary, columns=["metric", "count"])
    df_sum["share"] = df_sum["count"] / total if total > 0 else np.nan

    print("\n=== DIAGNOSE: EventId-Ebene (Open-Play-Flanken) ===")
    print(f"Events gesamt: {total}")
    print(df_sum)

    # Event-Report: relevante Spalten (robust)
    keep_cols = [
        "event_id",
        "match_id",
        "team_id",
        "_src_file",
        "time_source",
        "gametime_raw",
        "gametime_sec",
        "minute_of_play",
        "in_game_section",
        "time_na",
        "stoppage_time",
        "boundary_case",
        "outside_segment_range",
        "n_matching_segments",
        "not_assignable_despite_time",
        "ambiguous_multi_segment",
        "sonstige_probleme",
        "any_issue",
    ]

    # Falls manche Spalten fehlen (sollte nicht), nur vorhandene schreiben
    keep_cols = [c for c in keep_cols if c in ev2.columns]
    ev2[keep_cols].to_csv(OUT_EVENT, index=False)
    df_sum.to_csv(OUT_SUMMARY, index=False)

    print(f"\nEvent-Report gespeichert: {OUT_EVENT}")
    print(f"Summary gespeichert:     {OUT_SUMMARY}")

    # Zusätzlicher Fokus: Nachspielzeit-Events tabellarisch kurz ausgeben
    n_stop = int(ev2["stoppage_time"].sum())
    if n_stop > 0:
        ex = ev2.loc[ev2["stoppage_time"]].copy()
        ex = ex.sort_values(["match_id", "team_id", "minute_of_play"]).head(20)
        print("\nBeispiele Nachspielzeit (Top 20):")
        print(ex[["event_id", "match_id", "team_id", "in_game_section", "gametime_raw", "minute_of_play"]])

    # Falls time_na hoch ist, sofort Hinweis
    if int(ev2["time_na"].sum()) == total:
        print("\nHINWEIS: time_na=100% -> sehr wahrscheinlich GameTime-Parsing oder Spalte fehlt/leer.")
        print("Prüfe in den CSVs, ob GameTime existiert und Werte enthält (Format meist MMM:SS:CC).")


if __name__ == "__main__":
    main()
