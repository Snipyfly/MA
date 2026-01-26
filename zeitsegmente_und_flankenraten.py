# zeitsegmente_und_flankenraten.py
# -*- coding: utf-8 -*-

import os
import re
import gzip
import json
from pathlib import Path
from collections import deque, defaultdict

import requests
import xmltodict
import pandas as pd

# ============================================================
# KONFIG
# ============================================================

BASE_URL   = "https://httpget.distribution.production.datahub-sts.de/DeliveryPlatform/REST/PullOnce"
CUSTOMER_ID = "sgf-m9hk-4u2a-7i6w"
FEED_BASIC  = "DFL-03.05-EventData-Match-Basic_Extended"

TOKEN = os.getenv("DFL_API_TOKEN")

INPUT_DIR = Path("cleaned")  # hier liegen deine *_openplay.csv
CROSS_FILES_PATTERN = "flanken_*_openplay.csv"  # nur OpenPlay-Flanken
CACHE_DIR = INPUT_DIR / "_cache_basic_0305"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_MATCH_LEVEL = INPUT_DIR / "zeitsegmente_match_level.csv"
OUT_SEASON_LEAGUE = INPUT_DIR / "zeitsegmente_season_league.csv"

# ============================================================
# BASIS: HTTP + XML Helfer
# ============================================================

def http_get_xml(customer_id: str, feed: str, match_id: str, token: str | None = None) -> dict:
    url = f"{BASE_URL}/{customer_id}/{feed}/{match_id}"
    headers = {
        "Accept": "application/xml",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "MA-Zeitsegmente/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    data = resp.content
    if resp.headers.get("Content-Encoding", "").lower() == "gzip" or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)

    xml_text = data.decode("utf-8", errors="replace")
    return xmltodict.parse(xml_text)


def iter_all_keys(obj, key_name: str):
    q = deque([obj])
    while q:
        cur = q.popleft()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k == key_name:
                    yield v
                q.append(v)
        elif isinstance(cur, list):
            q.extend(cur)


def to_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


# ============================================================
# CACHING: 03.05 pro Match
# ============================================================

def cache_path_for_match(match_id: str) -> Path:
    return CACHE_DIR / f"{match_id}.json"


def get_events_basic(match_id: str) -> dict:
    """
    Holt 03.05 (EventsBasicExtended) und cached als JSON.
    """
    fp = cache_path_for_match(match_id)
    if fp.exists():
        with fp.open("r", encoding="utf-8") as f:
            return json.load(f)

    parsed = http_get_xml(CUSTOMER_ID, FEED_BASIC, match_id, TOKEN)
    pdr = parsed.get("PutDataRequest") or {}
    basic = pdr.get("EventsBasicExtended")
    if not basic:
        for val in iter_all_keys(parsed, "EventsBasicExtended"):
            basic = val
            break
    if not basic:
        raise ValueError("Kein 'EventsBasicExtended' in 03.05 gefunden.")

    with fp.open("w", encoding="utf-8") as f:
        json.dump(basic, f, ensure_ascii=False)

    return basic


# ============================================================
# MinuteOfPlay Parsing (inkl. Nachspielzeit)
# ============================================================

_MOP_RE = re.compile(r"^\s*(\d+)\.\s*(?:\+(\d+))?\s*$")

def parse_minute_of_play(mop: str | None) -> int | None:
    """
    DFL MinuteOfPlay Beispiele:
      "1." -> 1
      "45.+4" -> 49
      "90.+3" -> 93
    """
    if mop is None:
        return None
    mop = str(mop).strip()
    m = _MOP_RE.match(mop)
    if not m:
        return None
    base = int(m.group(1))
    add = int(m.group(2)) if m.group(2) is not None else 0
    return base + add


# ============================================================
# Spielzustand (Score) + Man-Advantage aus 03.05 ableiten
# ============================================================

def extract_teams_meta(basic: dict) -> dict:
    return {
        "match_id": basic.get("@MatchId"),
        "season_id": basic.get("@SeasonId"),
        "competition": basic.get("@Competition"),
        "competition_id": basic.get("@CompetitionId"),
        "home_id": basic.get("@HomeTeamId"),
        "home_name": basic.get("@HomeTeamName"),
        "guest_id": basic.get("@GuestTeamId"),
        "guest_name": basic.get("@GuestTeamName"),
    }


def extract_goal_events(basic: dict) -> list[dict]:
    """
    Tore aus ShotAtGoal mit @Result="x:y" (03.05).
    MinuteOfPlay gibt es i.d.R. am ShotAtGoal.
    """
    goals = []
    for node in iter_all_keys(basic, "ShotAtGoal"):
        for s in to_list(node):
            if not isinstance(s, dict):
                continue
            res = s.get("@Result")
            if not res:
                continue
            mop = parse_minute_of_play(s.get("@MinuteOfPlay"))
            if mop is None:
                continue
            try:
                sh, sg = res.split(":")
                sh, sg = int(sh), int(sg)
            except Exception:
                continue
            goals.append({"minute": mop, "score_home": sh, "score_guest": sg})
    goals.sort(key=lambda x: x["minute"])
    return goals


def extract_red_events(basic: dict) -> list[dict]:
    """
    Rote Karten aus CardColor=red/yellowred.
    MinuteOfPlay wird genutzt.
    """
    reds = []
    for node in iter_all_keys(basic, "Card"):
        for c in to_list(node):
            if not isinstance(c, dict):
                continue
            color = str(c.get("@CardColor", "")).lower()
            if color not in ("red", "yellowred"):
                continue
            mop = parse_minute_of_play(c.get("@MinuteOfPlay"))
            if mop is None:
                continue
            team_id = c.get("@TeamId")
            reds.append({"minute": mop, "team_id": team_id, "color": color})
    reds.sort(key=lambda x: x["minute"])
    return reds


def extract_match_end_minute(basic: dict) -> int:
    """
    Match-Ende als maximale MinuteOfPlay aus FinalWhistle (1. oder 2. Halbzeit).
    Fallback: @MinuteOfPlay am root (EventsBasicExtended).
    """
    mins = []

    # FinalWhistle in GameSection(s)
    for node in iter_all_keys(basic, "FinalWhistle"):
        for fw in to_list(node):
            if not isinstance(fw, dict):
                continue
            mop = parse_minute_of_play(fw.get("@MinuteOfPlay"))
            if mop is not None:
                mins.append(mop)

    # Root MinuteOfPlay
    mop_root = parse_minute_of_play(basic.get("@MinuteOfPlay"))
    if mop_root is not None:
        mins.append(mop_root)

    if not mins:
        # notfalls Standard 90
        return 90
    return max(mins)


def build_state_timelines(basic: dict) -> tuple[dict, dict]:
    """
    Liefert pro Minute m (1..end_min) den Score-State und Man-Adv-State
    für HOME und GUEST.

    Rückgabe:
      score_state[team_role][m] in {"leading","drawing","trailing"}
      man_state[team_role][m]   in {"adv","even","disadv"}
    """
    meta = extract_teams_meta(basic)
    home_id = meta["home_id"]
    guest_id = meta["guest_id"]

    goals = extract_goal_events(basic)
    reds  = extract_red_events(basic)
    end_min = extract_match_end_minute(basic)

    # -------- Score je Minute aufbauen --------
    # Start: 0:0 ab Minute 1
    score_home = 0
    score_guest = 0
    goal_idx = 0

    score_state = {"home": {}, "guest": {}}

    for m in range(1, end_min + 1):
        # alle Tore, die in Minute m gefallen sind, anwenden
        while goal_idx < len(goals) and goals[goal_idx]["minute"] == m:
            score_home = goals[goal_idx]["score_home"]
            score_guest = goals[goal_idx]["score_guest"]
            goal_idx += 1

        # Score-State für beide Teams
        if score_home > score_guest:
            score_state["home"][m] = "leading"
            score_state["guest"][m] = "trailing"
        elif score_home < score_guest:
            score_state["home"][m] = "trailing"
            score_state["guest"][m] = "leading"
        else:
            score_state["home"][m] = "drawing"
            score_state["guest"][m] = "drawing"

    # -------- Man-Adv je Minute aufbauen --------
    # Wir zählen rote Karten kumulativ.
    red_count = {home_id: 0, guest_id: 0}
    red_idx = 0

    man_state = {"home": {}, "guest": {}}

    for m in range(1, end_min + 1):
        while red_idx < len(reds) and reds[red_idx]["minute"] == m:
            tid = reds[red_idx]["team_id"]
            if tid in red_count:
                red_count[tid] += 1
            red_idx += 1

        # Spielerzahl relativ: mehr rote Karten = schlechter (disadv)
        # home vs guest
        if red_count[home_id] < red_count[guest_id]:
            man_state["home"][m] = "adv"
            man_state["guest"][m] = "disadv"
        elif red_count[home_id] > red_count[guest_id]:
            man_state["home"][m] = "disadv"
            man_state["guest"][m] = "adv"
        else:
            man_state["home"][m] = "even"
            man_state["guest"][m] = "even"

    return score_state, man_state


# ============================================================
# Segmentbildung: zusammenhängende Minuten gleicher Zustände
# ============================================================

def minutes_to_segments(match_id: str, team_id: str, team_role: str,
                        season_id: str, league_name: str,
                        score_state_map: dict, man_state_map: dict,
                        end_min: int) -> list[dict]:
    """
    Bildet Segmente: (start_min, end_min, minutes) pro Team.
    """
    segments = []

    cur_score = None
    cur_man = None
    seg_start = 1

    for m in range(1, end_min + 1):
        s = score_state_map[team_role][m]
        a = man_state_map[team_role][m]

        if cur_score is None:
            cur_score, cur_man = s, a
            seg_start = m
            continue

        if s != cur_score or a != cur_man:
            seg_end = m - 1
            segments.append({
                "match_id": match_id,
                "season": season_id,
                "league": league_name,
                "team_id": team_id,
                "team_role": team_role,
                "start_min": float(seg_start),
                "end_min": float(seg_end),
                "minutes": float(seg_end - seg_start + 1),
                "match_state_for_team": cur_score,
                "man_adv_state_for_team": cur_man,
            })
            cur_score, cur_man = s, a
            seg_start = m

    # letztes Segment
    if cur_score is not None:
        segments.append({
            "match_id": match_id,
            "season": season_id,
            "league": league_name,
            "team_id": team_id,
            "team_role": team_role,
            "start_min": float(seg_start),
            "end_min": float(end_min),
            "minutes": float(end_min - seg_start + 1),
            "match_state_for_team": cur_score,
            "man_adv_state_for_team": cur_man,
        })

    return segments


def build_segments_for_match(match_id: str) -> pd.DataFrame:
    basic = get_events_basic(match_id)
    meta = extract_teams_meta(basic)

    season_id = meta["season_id"]
    league_name = meta["competition"]

    end_min = extract_match_end_minute(basic)
    score_state, man_state = build_state_timelines(basic)

    home_id = meta["home_id"]
    guest_id = meta["guest_id"]

    seg_home = minutes_to_segments(match_id, home_id, "home", season_id, league_name,
                                   score_state, man_state, end_min)
    seg_guest = minutes_to_segments(match_id, guest_id, "guest", season_id, league_name,
                                    score_state, man_state, end_min)

    df = pd.DataFrame(seg_home + seg_guest)
    return df


# ============================================================
# Flanken: OpenPlay aus Files + korrekte Minute-Zuordnung
# ============================================================

def pick_match_id_column(df: pd.DataFrame) -> str | None:
    for c in ["MatchId", "SourceMatchId", "match_id"]:
        if c in df.columns:
            return c
    return None


def minute_from_gametime(gametime: str | float | int | None) -> int | None:
    """
    GameTime aus 03.06 sieht oft aus wie "006:38:95" (MM:SS:CC) oder "MM:SS".
    Wir mappen auf "angefangene Minute": floor(seconds/60)+1.
    """
    if gametime is None or (isinstance(gametime, float) and pd.isna(gametime)):
        return None
    if isinstance(gametime, (int, float)):
        sec = float(gametime)
    else:
        gt = str(gametime)
        parts = gt.split(":")
        try:
            if len(parts) == 3:
                mm, ss, cs = parts
                sec = int(mm) * 60 + int(ss) + int(cs) / 100.0
            elif len(parts) == 2:
                mm, ss = parts
                sec = int(mm) * 60 + float(ss)
            else:
                sec = float(gt)
        except Exception:
            return None

    if sec < 0:
        return None
    # KORREKT: angefangene Minute
    return int(sec // 60) + 1


def load_openplay_crosses() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob(CROSS_FILES_PATTERN))
    if not files:
        raise FileNotFoundError(f"Keine OpenPlay-Flanken-Dateien gefunden: {INPUT_DIR}/{CROSS_FILES_PATTERN}")

    print(f"Gefundene Flanken-Dateien: {len(files)}")

    all_df = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="latin-1")

        # Meta aus Dateiname: league + season
        # Beispiel: flanken_1_BL_DFL-SEA-0001K6_openplay.csv
        league = None
        if "_1_BL_" in fp.name:
            league = "Bundesliga"
        elif "_2_BL_" in fp.name:
            league = "2. Bundesliga"

        m_season = re.search(r"(DFL-SEA-[A-Z0-9\-]+)", fp.name)
        season = m_season.group(1) if m_season else None

        df["_league_file"] = league
        df["_season_file"] = season
        df["_file"] = fp.name

        all_df.append(df)

    out = pd.concat(all_df, ignore_index=True)

    # MatchId Spalte normalisieren
    match_col = pick_match_id_column(out)
    if not match_col:
        raise ValueError("Keine MatchId-Spalte in OpenPlay-Flanken gefunden (MatchId/SourceMatchId).")
    out = out.rename(columns={match_col: "match_id"})

    if "TeamId" not in out.columns:
        raise ValueError("OpenPlay-Flanken haben keine TeamId-Spalte.")
    out = out.rename(columns={"TeamId": "team_id"})

    # Minute korrekt berechnen (angefangene Minute)
    if "GameTime" not in out.columns:
        raise ValueError("OpenPlay-Flanken haben keine GameTime-Spalte.")
    out["minute"] = out["GameTime"].apply(minute_from_gametime)

    return out


# ============================================================
# Flanken auf Segmente zählen
# ============================================================

def count_crosses_into_segments(df_seg: pd.DataFrame, df_cross: pd.DataFrame) -> pd.DataFrame:
    """
    Zählt Flanken je Segment (match_id, team_id, start_min..end_min) anhand df_cross["minute"].
    """
    # Safety: sortiert und keine NAs
    df_cross2 = df_cross.dropna(subset=["match_id", "team_id", "minute"]).copy()
    df_cross2["minute"] = df_cross2["minute"].astype(int)

    # Für Performance: pro Match/Team Minutenlisten bauen
    minutes_by_mt = defaultdict(list)
    for (mid, tid), g in df_cross2.groupby(["match_id", "team_id"]):
        minutes_by_mt[(mid, tid)] = g["minute"].tolist()

    crosses_counts = []
    for i, row in df_seg.iterrows():
        mid = row["match_id"]
        tid = row["team_id"]
        s = int(row["start_min"])
        e = int(row["end_min"])
        mins = minutes_by_mt.get((mid, tid), [])
        # count minutes in [s,e]
        c = sum(1 for m in mins if s <= m <= e)
        crosses_counts.append(c)

    df_seg = df_seg.copy()
    df_seg["crosses_openplay"] = crosses_counts
    return df_seg


# ============================================================
# Aggregation
# ============================================================

def aggregate_season_league(df_match_level: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert auf season/league/state/adv:
      matches, teams, minutes, crosses, mean_crosses_per_90, sd_crosses_per_90
    """
    # pro Team-Match-Block -> crosses per 90
    # (damit SD sinnvoll ist)
    tm = (
        df_match_level.groupby(
            ["match_id", "team_id", "season", "league", "match_state_for_team", "man_adv_state_for_team"],
            dropna=False
        )[["minutes", "crosses_openplay"]]
        .sum()
        .reset_index()
    )
    tm["crosses_per_90"] = tm["crosses_openplay"] / tm["minutes"] * 90.0

    agg = (
        tm.groupby(["season", "league", "match_state_for_team", "man_adv_state_for_team"], dropna=False)
        .agg(
            matches=("match_id", "nunique"),
            teams=("team_id", "nunique"),
            minutes=("minutes", "sum"),
            crosses=("crosses_openplay", "sum"),
            mean_crosses_per_90=("crosses_per_90", "mean"),
            sd_crosses_per_90=("crosses_per_90", "std"),
        )
        .reset_index()
    )
    return agg


# ============================================================
# MAIN
# ============================================================

def main():
    # 1) OpenPlay-Flanken laden
    df_cross = load_openplay_crosses()
    unique_matches = df_cross["match_id"].nunique()
    print(f"Unique Matches in deinen OpenPlay-Flanken: {unique_matches}")

    match_ids = sorted(df_cross["match_id"].dropna().unique().tolist())

    # 2) Segmente pro Match bauen
    all_seg = []
    failed = 0

    for idx, mid in enumerate(match_ids, start=1):
        if idx % 50 == 0 or idx == 1:
            print(f"[{idx}/{len(match_ids)}] Segmente: {mid}")
        try:
            seg = build_segments_for_match(mid)
            all_seg.append(seg)
        except Exception as e:
            failed += 1
            print(f"  !! Fehler bei {mid}: {e}")

    if not all_seg:
        raise RuntimeError("Keine Segmente gebaut.")

    df_seg_all = pd.concat(all_seg, ignore_index=True)

    # 3) Flanken in Segmente zählen
    df_seg_all = count_crosses_into_segments(df_seg_all, df_cross)

    # 4) Speichern Match-Level
    df_seg_all.to_csv(OUT_MATCH_LEVEL, index=False)
    print(f"Match-Level Segmente gespeichert: {OUT_MATCH_LEVEL}")

    # 5) Aggregation Season/League
    df_agg = aggregate_season_league(df_seg_all)
    df_agg.to_csv(OUT_SEASON_LEAGUE, index=False)
    print(f"Season/League Aggregation gespeichert: {OUT_SEASON_LEAGUE}")

    print(f"Fertige Matches: {len(match_ids) - failed} | Fehlgeschlagen: {failed}")
    print(f"Cache-Files im Ordner: {CACHE_DIR} (wird bei erneutem Run wiederverwendet)")


if __name__ == "__main__":
    main()
