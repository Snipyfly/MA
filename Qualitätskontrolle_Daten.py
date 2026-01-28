# Qualitätskontrolle_Daten.py
from __future__ import annotations

import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from f_zeitsegmente_und_flankenraten import (
    extract_goal_events,
    extract_red_events,
    extract_teams_meta,
    get_events_basic,
)

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
OUT_MISSING_SUMMARY = CLEANED_DIR / "j_missing_data_summary.csv"
OUT_MISSING_EVENTS = CLEANED_DIR / "j_missing_data_events.csv"
OUT_SEGMENT_CHECK = CLEANED_DIR / "g_check_flanken_vs_segmente_report.csv"
OUT_DRAWING_EVEN = CLEANED_DIR / "k_missing_drawing_even_details.csv"


# ============================================================
# HELFER
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

    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)

    parts = s.split(":")
    try:
        if len(parts) == 3:
            m = int(parts[0])
            sec = int(parts[1])
            cc = int(parts[2])
            return m * 60 + sec + cc / 100.0
        if len(parts) == 2:
            m = int(parts[0])
            sec = int(parts[1])
            return m * 60 + sec
        return np.nan
    except Exception:
        return np.nan


def seconds_to_minute_of_play(sec: float) -> float:
    """
    Minute-of-play (1-basiert) aus Sekunden:
      0.00s .. 59.99s -> Minute 1
      60.00s .. 119.99s -> Minute 2
    """
    if pd.isna(sec) or sec < 0:
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
            "Erwartet wird deine Match-Level Segmentdatei aus zeitsegmente_und_flankenraten.py."
        )

    seg = safe_read_csv(SEGMENTS_FP)

    match_col = pick_first_existing(seg, ["match_id", "MatchId"])
    team_col = pick_first_existing(seg, ["team_id", "TeamId"])
    if match_col is None or team_col is None:
        raise ValueError("Segmentdatei braucht match_id und team_id (oder MatchId/TeamId).")

    start_col = pick_first_existing(seg, ["start_mop", "start_minute_of_play", "start_min", "start_minute"])
    end_col = pick_first_existing(seg, ["end_mop", "end_minute_of_play", "end_min", "end_minute"])

    if start_col is None or end_col is None:
        raise ValueError(
            "Segmentdatei braucht Start/Ende-Spalten, z.B. start_mop/end_mop "
            "oder start_minute_of_play/end_minute_of_play."
        )

    seg = seg.rename(
        columns={
            match_col: "match_id",
            team_col: "team_id",
            start_col: "seg_start_mop",
            end_col: "seg_end_mop",
        }
    )

    seg["seg_start_mop"] = pd.to_numeric(seg["seg_start_mop"], errors="coerce")
    seg["seg_end_mop"] = pd.to_numeric(seg["seg_end_mop"], errors="coerce")

    return seg.dropna(subset=["match_id", "team_id", "seg_start_mop", "seg_end_mop"])


# ============================================================
# FLANKEN-EVENTS LADEN + NORMALISIEREN
# ============================================================

def load_cross_events() -> pd.DataFrame:
    fps = []
    for pat in CROSS_PATTERNS:
        fps.extend(sorted(CLEANED_DIR.glob(pat)))

    if not fps:
        raise FileNotFoundError(
            f"Keine Flanken-OpenPlay-Dateien gefunden in {CLEANED_DIR} mit {CROSS_PATTERNS}"
        )

    frames = []
    for fp in fps:
        df = safe_read_csv(fp)
        df["_src_file"] = fp.name
        frames.append(df)

    c = pd.concat(frames, ignore_index=True)

    match_col = pick_first_existing(c, ["MatchId", "match_id", "SourceMatchId"])
    team_col = pick_first_existing(c, ["TeamId", "team_id", "CrossTeamId"])
    event_col = pick_first_existing(c, ["EventId", "event_id", "CrossEventId"])
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

    if gametime_col is not None:
        c["time_source"] = "GameTime"
        c["gametime_raw"] = c[gametime_col]
        c["gametime_sec"] = c[gametime_col].map(parse_dfl_gametime_to_seconds)
    else:
        c["time_source"] = "NONE"
        c["gametime_raw"] = pd.NA
        c["gametime_sec"] = np.nan

    if ingame_section_col is not None:
        c["in_game_section"] = c[ingame_section_col]
    else:
        c["in_game_section"] = pd.NA

    c["minute_of_play"] = c["gametime_sec"].map(seconds_to_minute_of_play)

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
    rng = (
        seg.groupby(["match_id", "team_id"], as_index=False)
        .agg(seg_min_start=("seg_start_mop", "min"), seg_max_end=("seg_end_mop", "max"))
    )

    ev = events.merge(rng, on=["match_id", "team_id"], how="left")

    ev["outside_segment_range"] = False
    mask_has_rng = ev["seg_min_start"].notna() & ev["seg_max_end"].notna() & ev["minute_of_play"].notna()
    ev.loc[mask_has_rng, "outside_segment_range"] = (
        (ev.loc[mask_has_rng, "minute_of_play"] < ev.loc[mask_has_rng, "seg_min_start"])
        | (ev.loc[mask_has_rng, "minute_of_play"] > ev.loc[mask_has_rng, "seg_max_end"])
    )

    hit_counts = []
    seg_group = {k: g for k, g in seg.groupby(["match_id", "team_id"])}

    for _, row in ev[["match_id", "team_id", "minute_of_play"]].iterrows():
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

    ev["sonstige_probleme"] = False
    ev.loc[
        ev["minute_of_play"].notna() & ev["seg_min_start"].notna() & ev["n_matching_segments"].isna(),
        "sonstige_probleme",
    ] = True

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
# CHECK: Flanken vs Segmente
# ============================================================

def check_flanken_vs_segmente() -> None:
    if not SEGMENTS_FP.exists():
        raise FileNotFoundError(f"Nicht gefunden: {SEGMENTS_FP}")

    seg = safe_read_csv(SEGMENTS_FP)

    seg_match_col = pick_first_existing(seg, ["match_id", "MatchId", "SourceMatchId"])
    seg_team_col = pick_first_existing(seg, ["team_id", "TeamId"])
    seg_cross_col = pick_first_existing(seg, ["crosses_openplay", "crosses", "openplay_crosses"])

    missing = [x for x in [seg_match_col, seg_team_col, seg_cross_col] if x is None]
    if missing:
        raise ValueError(
            "Zeitsegmente fehlen erforderliche Spalten. "
            f"Gefunden: match={seg_match_col}, team={seg_team_col}, crosses={seg_cross_col}. "
            f"Spalten im File: {list(seg.columns)}"
        )

    seg_team_match = (
        seg.groupby([seg_match_col, seg_team_col], dropna=False)[seg_cross_col]
        .sum()
        .reset_index()
        .rename(
            columns={
                seg_match_col: "match_id",
                seg_team_col: "team_id",
                seg_cross_col: "crosses_openplay_from_segments",
            }
        )
    )

    cross_files = []
    for pat in ["flanken_*_openplay.csv"]:
        cross_files.extend(sorted(CLEANED_DIR.glob(pat)))

    if not cross_files:
        raise FileNotFoundError(
            f"Keine Dateien gefunden in {CLEANED_DIR} mit Pattern ['flanken_*_openplay.csv']"
        )

    dfs = []
    for fp in cross_files:
        df = safe_read_csv(fp)
        df["_file"] = fp.name
        dfs.append(df)

    crosses = pd.concat(dfs, ignore_index=True)

    cross_match_col = pick_first_existing(crosses, ["MatchId", "SourceMatchId", "match_id"])
    cross_team_col = pick_first_existing(crosses, ["TeamId", "team_id"])
    cross_event_col = pick_first_existing(crosses, ["EventId", "event_id"])

    if cross_match_col is None or cross_team_col is None:
        raise ValueError(
            "In den OpenPlay-Flanken fehlen MatchId/TeamId. "
            f"Gefunden: match={cross_match_col}, team={cross_team_col}. "
            f"Spalten: {list(crosses.columns)}"
        )

    if cross_event_col is not None:
        cross_team_match = (
            crosses.dropna(subset=[cross_match_col, cross_team_col])
            .assign(_eid=crosses[cross_event_col].astype(str))
            .groupby([cross_match_col, cross_team_col])["_eid"]
            .nunique()
            .reset_index()
            .rename(
                columns={
                    cross_match_col: "match_id",
                    cross_team_col: "team_id",
                    "_eid": "crosses_openplay_from_files",
                }
            )
        )
    else:
        cross_team_match = (
            crosses.dropna(subset=[cross_match_col, cross_team_col])
            .groupby([cross_match_col, cross_team_col])
            .size()
            .reset_index(name="crosses_openplay_from_files")
            .rename(columns={cross_match_col: "match_id", cross_team_col: "team_id"})
        )

    chk = seg_team_match.merge(
        cross_team_match,
        on=["match_id", "team_id"],
        how="outer",
        indicator=True,
    )

    chk["crosses_openplay_from_segments"] = chk["crosses_openplay_from_segments"].fillna(0).astype(int)
    chk["crosses_openplay_from_files"] = chk["crosses_openplay_from_files"].fillna(0).astype(int)
    chk["diff"] = chk["crosses_openplay_from_segments"] - chk["crosses_openplay_from_files"]

    total_pairs = len(chk)
    ok_pairs = (chk["diff"] == 0).sum()
    bad_pairs = (chk["diff"] != 0).sum()

    missing_in_segments = (chk["_merge"] == "right_only").sum()
    missing_in_files = (chk["_merge"] == "left_only").sum()

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
        print(
            ex[
                [
                    "match_id",
                    "team_id",
                    "crosses_openplay_from_files",
                    "crosses_openplay_from_segments",
                    "diff",
                    "_merge",
                ]
            ].to_string(index=False)
        )

    chk.to_csv(OUT_SEGMENT_CHECK, index=False)
    print(f"\nDetailreport gespeichert: {OUT_SEGMENT_CHECK}")


# ============================================================
# CHECK: Missing Matches + drawing/even
# ============================================================

def check_missing_matches() -> None:
    flank_files = sorted(CLEANED_DIR.glob("flanken_*_openplay.csv"))
    if not flank_files:
        raise FileNotFoundError("Keine flanken_*_openplay.csv in cleaned gefunden.")

    all_flank_mids = set()
    for fp in flank_files:
        df = pd.read_csv(fp)
        if "MatchId" not in df.columns:
            raise ValueError(f"{fp.name}: MatchId-Spalte fehlt.")
        all_flank_mids |= set(df["MatchId"].dropna().astype(str).unique())

    print("Unique Matches in OpenPlay-Flanken:", len(all_flank_mids))

    seg_fp = CLEANED_DIR / "f_zeitsegmente_match_level.csv"
    df_seg = pd.read_csv(seg_fp)

    match_col = None
    for c in ["match_id", "MatchId", "matchId"]:
        if c in df_seg.columns:
            match_col = c
            break
    if match_col is None:
        raise ValueError(f"Keine MatchId-Spalte in {seg_fp.name} gefunden. Spalten: {list(df_seg.columns)}")

    seg_mids = set(df_seg[match_col].dropna().astype(str).unique())
    print("Unique Matches in Zeitsegmenten:", len(seg_mids))

    missing = sorted(all_flank_mids - seg_mids)
    extra = sorted(seg_mids - all_flank_mids)

    print("\nFehlende Matches (in Flanken, aber nicht in Segmenten):", len(missing))
    print("Beispiel:", missing[:20])

    print("\nZusätzliche Matches (in Segmenten, aber nicht in Flanken):", len(extra))
    print("Beispiel:", extra[:20])

    required_cols = {"match_state_for_team", "man_adv_state_for_team"}
    if required_cols.issubset(df_seg.columns):
        df_seg = df_seg.assign(
            _mid=df_seg[match_col].astype(str),
            _is_drawing_even=(
                (df_seg["match_state_for_team"] == "drawing")
                & (df_seg["man_adv_state_for_team"] == "even")
            ),
        )
        has_drawing_even = df_seg.groupby("_mid")["_is_drawing_even"].any()
        missing_drawing_even = sorted(has_drawing_even[~has_drawing_even].index.tolist())
        print("\nMatches OHNE drawing/even Segment:", len(missing_drawing_even))
        print("Beispiel:", missing_drawing_even[:20])
    else:
        print(
            "\nHinweis: match_state_for_team / man_adv_state_for_team fehlen in "
            "Segmentdatei – drawing/even Check übersprungen."
        )


# ============================================================
# CHECK: Diagnose Assignment + Summary
# ============================================================

def diagnose_event_assignment() -> pd.DataFrame:
    seg = load_segments()
    ev = load_cross_events()

    ev2 = diagnose_assignment(ev, seg)

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

    keep_cols = [c for c in keep_cols if c in ev2.columns]
    ev2[keep_cols].to_csv(OUT_EVENT, index=False)
    df_sum.to_csv(OUT_SUMMARY, index=False)

    print(f"\nEvent-Report gespeichert: {OUT_EVENT}")
    print(f"Summary gespeichert:     {OUT_SUMMARY}")

    n_stop = int(ev2["stoppage_time"].sum())
    if n_stop > 0:
        ex = ev2.loc[ev2["stoppage_time"]].copy()
        ex = ex.sort_values(["match_id", "team_id", "minute_of_play"]).head(20)
        print("\nBeispiele Nachspielzeit (Top 20):")
        print(ex[["event_id", "match_id", "team_id", "in_game_section", "gametime_raw", "minute_of_play"]])

    if int(ev2["time_na"].sum()) == total:
        print("\nHINWEIS: time_na=100% -> sehr wahrscheinlich GameTime-Parsing oder Spalte fehlt/leer.")
        print("Prüfe in den CSVs, ob GameTime existiert und Werte enthält (Format meist MMM:SS:CC).")

    return ev2


# ============================================================
# CHECK: Missing data summary
# ============================================================

def check_missing_data(ev2: pd.DataFrame) -> None:
    total = len(ev2)
    summary = [
        ("total_openplay_crosses_events", total),
        ("time_na", int(ev2["time_na"].sum())),
        ("outside_segment_range", int(ev2["outside_segment_range"].sum())),
        ("not_assignable_despite_time", int(ev2["not_assignable_despite_time"].sum())),
        ("ambiguous_multi_segment", int(ev2["ambiguous_multi_segment"].sum())),
    ]
    df_sum = pd.DataFrame(summary, columns=["metric", "count"])
    df_sum["share"] = df_sum["count"] / total if total > 0 else pd.NA

    print("\n=== CHECK: Fehlende/inkonsistente Daten (Open-Play-Flanken) ===")
    print(df_sum)

    missing_mask = (
        ev2["time_na"]
        | ev2["outside_segment_range"]
        | ev2["not_assignable_despite_time"]
        | ev2["ambiguous_multi_segment"]
    )

    if missing_mask.any():
        examples = ev2.loc[missing_mask].copy()
        cols = [
            "event_id",
            "match_id",
            "team_id",
            "minute_of_play",
            "in_game_section",
            "time_na",
            "outside_segment_range",
            "not_assignable_despite_time",
            "ambiguous_multi_segment",
        ]
        cols = [c for c in cols if c in examples.columns]
        print("\nBeispiele fehlende/inkonsistente Daten (Top 20):")
        print(examples[cols].head(20).to_string(index=False))
        examples.to_csv(OUT_MISSING_EVENTS, index=False)
        print(f"Detail-Export gespeichert: {OUT_MISSING_EVENTS}")
    else:
        print("\nKeine fehlenden/inkonsistenten Events gefunden.")

    OUT_MISSING_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(OUT_MISSING_SUMMARY, index=False)
    print(f"\nSummary gespeichert: {OUT_MISSING_SUMMARY}")


# ============================================================
# CHECK: drawing/even Detailanalyse
# ============================================================

def check_missing_drawing_even_details() -> None:
    if not SEGMENTS_FP.exists():
        raise FileNotFoundError(f"Segment-Datei nicht gefunden: {SEGMENTS_FP}")

    seg = pd.read_csv(SEGMENTS_FP)

    required_cols = {"match_state_for_team", "man_adv_state_for_team"}
    if not required_cols.issubset(seg.columns):
        raise ValueError(
            "Segmentdatei braucht match_state_for_team und man_adv_state_for_team "
            f"für den drawing/even Check. Spalten: {list(seg.columns)}"
        )

    match_col = None
    for c in ["match_id", "MatchId", "matchId"]:
        if c in seg.columns:
            match_col = c
            break
    if match_col is None:
        raise ValueError(f"Keine MatchId-Spalte in {SEGMENTS_FP.name} gefunden. Spalten: {list(seg.columns)}")

    seg = seg.assign(
        _mid=seg[match_col].astype(str),
        _is_drawing_even=(
            (seg["match_state_for_team"] == "drawing")
            & (seg["man_adv_state_for_team"] == "even")
        ),
    )
    has_drawing_even = seg.groupby("_mid")["_is_drawing_even"].any()

    missing = sorted(has_drawing_even[~has_drawing_even].index.tolist())
    print("Matches OHNE drawing/even Segment:", len(missing))
    print("Beispiel:", missing[:20])

    if not missing:
        print("Keine betroffenen Matches gefunden.")
        return

    rows = []
    for mid in missing:
        try:
            basic = get_events_basic(mid)
            meta = extract_teams_meta(basic)
            goals = extract_goal_events(basic)
            reds = extract_red_events(basic)

            first_goal = goals[0]["minute"] if goals else pd.NA
            first_red = reds[0]["minute"] if reds else pd.NA
            kickoff = basic.get("@Kickoff") or basic.get("@MatchDate") or basic.get("@Date")

            rows.append(
                {
                    "match_id": mid,
                    "home_team": meta.get("home_name"),
                    "away_team": meta.get("guest_name"),
                    "match_date": kickoff,
                    "goals_count": len(goals),
                    "reds_count": len(reds),
                    "first_goal_minute": first_goal,
                    "first_red_minute": first_red,
                    "goal_in_minute_1": bool(goals and goals[0]["minute"] == 1),
                    "red_in_minute_1": bool(reds and reds[0]["minute"] == 1),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "match_id": mid,
                    "home_team": pd.NA,
                    "away_team": pd.NA,
                    "match_date": pd.NA,
                    "goals_count": pd.NA,
                    "reds_count": pd.NA,
                    "first_goal_minute": pd.NA,
                    "first_red_minute": pd.NA,
                    "goal_in_minute_1": pd.NA,
                    "red_in_minute_1": pd.NA,
                    "error": str(exc),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DRAWING_EVEN, index=False)
    print(f"Detail-Export gespeichert: {OUT_DRAWING_EVEN}")
    print(df.head(20).to_string(index=False))


# ============================================================
# RUNNER
# ============================================================

def run_step(name: str, func) -> None:
    print("\n" + "=" * 80)
    print(f"START: {name}")
    print("=" * 80)
    try:
        func()
        print(f"END: {name}")
    except Exception as exc:
        print(f"FEHLER in {name}: {exc}")
        print(traceback.format_exc())


def main() -> None:
    run_step("g_checks_zeitsegmente", check_flanken_vs_segmente)
    run_step("h_check_missing_matches", check_missing_matches)
    ev2_holder: dict[str, pd.DataFrame] = {}

    def run_diagnose() -> None:
        ev2_holder["df"] = diagnose_event_assignment()

    run_step("i_diagnose_event_assignment", run_diagnose)
    run_step(
        "j_missing_data_check",
        lambda: check_missing_data(ev2_holder.get("df", diagnose_event_assignment())),
    )
    run_step("k_check_missing_drawing_even_details", check_missing_drawing_even_details)


if __name__ == "__main__":
    main()
