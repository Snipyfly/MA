# MA Datenimport.py

import os
import gzip
from collections import deque

import requests
import xmltodict
import pandas as pd

from a_season_scanner_bl_bl2 import get_match_ids_last_5_seasons

# ============================================================
# KONFIGURATION
# ============================================================

BASE_URL = "https://httpget.distribution.production.datahub-sts.de/DeliveryPlatform/REST/PullOnce"
CUSTOMER_ID = "sgf-m9hk-4u2a-7i6w"

FEED_ADVANCED = "DFL-03.06-ErweiterteEreignisse-Spiel-Roh"
FEED_BASIC = "DFL-03.05-EventData-Match-Basic_Extended"

TOKEN = os.getenv("DFL_API_TOKEN")  # oder direkt als String eintragen

# ============================================================
# BASIS-FUNKTIONEN: Abruf + XML-Helfer
# ============================================================

def http_get_xml(customer_id, feed, match_id, token=None):
    """
    Ruft einen DFL-PullOnce-Feed ab und gibt ihn als dict (xmltodict) zurück.
    """
    url = f"{BASE_URL}/{customer_id}/{feed}/{match_id}"
    headers = {
        "Accept": "application/xml",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "MA-Flankenanalyse/1.0",
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


def iter_all_keys(obj, key_name):
    """
    Generator: liefert ALLE Werte für key_name im gesamten verschachtelten Objekt.
    """
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
# XML -> DataFrame: Plays (inkl. Flanken)
# ============================================================

def plays_to_df(plays):
    """
    Wandelt eine Liste von <Play>-Knoten (AdvancedEvents) in ein DataFrame.
    """
    rows = []
    for p in plays:
        if not isinstance(p, dict):
            continue
        rows.append({
            "EventId":            p.get("@EventId"),
            "TeamId":             p.get("@TeamId"),
            "PlayerId":           p.get("@PlayerId"),
            "Evaluation":         p.get("@Evaluation"),
            "IsPass":             p.get("@IsPass"),
            "IsCross":            p.get("@IsCross"),
            "IsCorner":           p.get("@IsCorner"),
            "IsFreeKick":         p.get("@IsFreeKick"),
            "IsGoalKick":         p.get("@IsGoalKick"),
            "IsThrowIn":          p.get("@IsThrowIn"),
            "IsKickOff":          p.get("@IsKickOff"),
            "InGameSection":      p.get("@InGameSection") or p.get("@GameSection"),
            "ReceiverId":         p.get("@ReceiverId"),
            "ReceptionId":        p.get("@ReceptionId"),
            "PlayNumInPossession":p.get("@PlayNumInPossession"),
            "ThroughBall":        p.get("@ThroughBall"),
            "DefensiveState":     p.get("@DefensiveState"),
            "SyncedEventTime":    p.get("@SyncedEventTime"),
            "GameTime":           p.get("@GameTime"),
            "X":                  p.get("@X-Position"),
            "Y":                  p.get("@Y-Position"),
            "X_dir":              p.get("@X-Direction"),
            "Y_dir":              p.get("@Y-Direction"),
            "X_rec":              p.get("@X-PositionReceiver"),
            "Y_rec":              p.get("@Y-PositionReceiver"),
            "MaxHeight":          p.get("@MaxHeight"),
            "Distance":           p.get("@Distance"),
            "PressureOnReceiver": p.get("@PressureOnReceiver"),
            "X_PlayerSpeed":      p.get("@X-PlayerSpeed"),
            "Y_PlayerSpeed":      p.get("@Y-PlayerSpeed"),
            "NumAttPlayersAhead": p.get("@NumAttackingPlayersAhead"),
            "NumDefInBox":        p.get("@NumDefendingPlayersInBox"),
            "NumAttInBox":        p.get("@NumAttackingPlayersInBox"),
        })
    df = pd.DataFrame(rows)

    # Booleans normalisieren
    for col in ["IsPass", "IsCross", "IsCorner", "IsFreeKick", "IsGoalKick", "IsThrowIn", "IsKickOff"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(pd.NA)

    # Numerische Felder
    num_cols = [
        "X", "Y", "X_dir", "Y_dir", "X_rec", "Y_rec",
        "MaxHeight", "Distance",
        "X_PlayerSpeed", "Y_PlayerSpeed",
        "NumAttPlayersAhead", "NumDefInBox", "NumAttInBox"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ============================================================
# XML -> DataFrame: ShotAtGoal (Schüsse mit xG)
# ============================================================

def shots_to_df(shots):
    """
    Wandelt eine Liste von <ShotAtGoal>-Knoten in ein DataFrame.
    """
    rows = []
    for s in shots:
        if not isinstance(s, dict):
            continue
        rows.append({
            "EventId":                  s.get("@EventId"),
            "TeamId":                   s.get("@TeamId"),
            "PlayerId":                 s.get("@PlayerId"),
            "IsPenalty":                s.get("@IsPenalty"),
            "IsFreeKick":               s.get("@IsFreeKick"),
            "IsCorner":                 s.get("@IsCorner"),
            "SyncSuccessful":           s.get("@SyncSuccessful"),
            "SyncedEventTime":          s.get("@SyncedEventTime"),
            "GameTime":                 s.get("@GameTime"),
            "X":                        s.get("@X-Position"),
            "Y":                        s.get("@Y-Position"),
            "SyncedFrameId":            s.get("@SyncedFrameId"),
            "xG":                       s.get("@xG"),
            "AngleToGoal":              s.get("@AngleToGoal"),
            "DistanceToGoal":           s.get("@DistanceToGoal"),
            "PressureOnPlayer":         s.get("@PressureOnPlayer"),
            "X_PlayerSpeed":            s.get("@X-PlayerSpeed"),
            "Y_PlayerSpeed":            s.get("@Y-PlayerSpeed"),
            "DistanceGoalkeeperToGoal": s.get("@DistanceGoalkeeperToGoal"),
            "X_PositionGoalkeeper":     s.get("@X-PositionGoalkeeper"),
            "Y_PositionGoalkeeper":     s.get("@Y-PositionGoalkeeper"),
            "ShotResult":               s.get("@ShotResult"),
            "DefensiveState":           s.get("@DefensiveState"),
            "InGameSection":            s.get("@InGameSection"),
            "NumDefInBox":              s.get("@NumDefendingPlayersInBox"),
            "NumAttInBox":              s.get("@NumAttackingPlayersInBox"),
            "NumDefendersInShotLane":   s.get("@NumDefendersInShotLane"),
        })
    df = pd.DataFrame(rows)

    # Booleans normalisieren
    for col in ["IsPenalty", "IsFreeKick", "IsCorner", "SyncSuccessful"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(pd.NA)

    # Numerische Felder
    num_cols = [
        "X", "Y", "xG", "AngleToGoal", "DistanceToGoal",
        "PressureOnPlayer", "X_PlayerSpeed", "Y_PlayerSpeed",
        "DistanceGoalkeeperToGoal", "X_PositionGoalkeeper", "Y_PositionGoalkeeper",
        "NumDefInBox", "NumAttInBox", "NumDefendersInShotLane"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ============================================================
# Plays & Shots pro Match (03.06)
# ============================================================

def get_advanced_for_match(match_id):
    """
    Holt den AdvancedEvents-Knoten für ein Match (03.06).
    """
    parsed = http_get_xml(CUSTOMER_ID, FEED_ADVANCED, match_id, TOKEN)
    pdr = parsed.get("PutDataRequest") or {}
    advanced = pdr.get("AdvancedEvents")
    if not advanced:
        for val in iter_all_keys(parsed, "AdvancedEvents"):
            advanced = val
            break
    if not advanced:
        raise ValueError("Kein 'AdvancedEvents'-Knoten im Feed 03.06 gefunden.")
    return advanced


def get_plays_for_match(match_id):
    advanced = get_advanced_for_match(match_id)
    all_play_nodes = []
    for node in iter_all_keys(advanced, "Play"):
        all_play_nodes.extend(to_list(node))
    df = plays_to_df(all_play_nodes)
    df["MatchId"] = match_id
    return df


def get_shots_for_match(match_id):
    advanced = get_advanced_for_match(match_id)
    all_shot_nodes = []
    for node in iter_all_keys(advanced, "ShotAtGoal"):
        all_shot_nodes.extend(to_list(node))
    df = shots_to_df(all_shot_nodes)
    df["MatchId"] = match_id
    return df


def get_crosses_for_match(match_id):
    plays = get_plays_for_match(match_id)
    if "IsCross" in plays.columns:
        crosses = plays[plays["IsCross"] == True].copy()
    else:
        crosses = plays[plays["Evaluation"].astype(str).str.contains("cross", case=False, na=False)].copy()
    return crosses

# ============================================================
# BASIC-Events (03.05) für Spielstand & rote Karten
# ============================================================

def get_match_basic(match_id):
    """
    Holt EventsBasicExtended (03.05) für ein Match.
    """
    parsed = http_get_xml(CUSTOMER_ID, FEED_BASIC, match_id, TOKEN)
    pdr = parsed.get("PutDataRequest") or {}
    basic = pdr.get("EventsBasicExtended")
    if not basic:
        for val in iter_all_keys(parsed, "EventsBasicExtended"):
            basic = val
            break
    if not basic:
        raise ValueError("Kein 'EventsBasicExtended'-Knoten im Feed 03.05 gefunden.")
    return basic


def build_score_and_redcard_timeline(basic):
    """
    Aus EventsBasicExtended:
    - Score-Timeline aus ShotAtGoal-Events mit Result="x:y"
    - Red-Card-Timeline aus Card-Events mit CardColor=red oder yellowRed
    """
    # Meta
    meta = {
        "MatchId":       basic.get("@MatchId"),
        "HomeTeamId":    basic.get("@HomeTeamId"),
        "GuestTeamId":   basic.get("@GuestTeamId"),
        "HomeTeamName":  basic.get("@HomeTeamName"),
        "GuestTeamName": basic.get("@GuestTeamName"),
    }

    # --------------------------------------------------------
    # Tore / Zwischenstände (ShotAtGoal mit Result="x:y")
    # --------------------------------------------------------
    goal_rows = []
    for node in iter_all_keys(basic, "ShotAtGoal"):
        for s in to_list(node):
            if not isinstance(s, dict):
                continue
            res = s.get("@Result")
            if not res:
                continue
            try:
                sh, sg = res.split(":")
                sh = int(sh)
                sg = int(sg)
            except Exception:
                continue

            goal_rows.append({
                "EventTime": s.get("@EventTime"),
                "ScoreHome": sh,
                "ScoreGuest": sg,
            })

    goals_df = pd.DataFrame(goal_rows)
    if not goals_df.empty:
        goals_df["EventTime"] = pd.to_datetime(
            goals_df["EventTime"],
            utc=True,
            errors="coerce"
        ).dt.tz_localize(None)
        goals_df = goals_df.dropna(subset=["EventTime"]).sort_values("EventTime")

    # --------------------------------------------------------
    # Rote Karten (CardColor = red / yellowRed)
    # --------------------------------------------------------
    red_rows = []
    for node in iter_all_keys(basic, "Card"):
        for c in to_list(node):
            if not isinstance(c, dict):
                continue
            color = c.get("@CardColor", "").lower()
            if color in ("red", "yellowred"):
                red_rows.append({
                    "EventTime": c.get("@EventTime"),
                    "TeamId":    c.get("@TeamId"),
                    "CardColor": color,
                })

    red_df = pd.DataFrame(red_rows)
    if not red_df.empty:
        red_df["EventTime"] = pd.to_datetime(
            red_df["EventTime"],
            utc=True,
            errors="coerce"
        ).dt.tz_localize(None)
        red_df = red_df.dropna(subset=["EventTime"]).sort_values("EventTime")

    return goals_df, red_df, meta

# ============================================================
# GameTime-Parsing
# ============================================================

def gametime_to_seconds(gt):
    """
    Wandelt GameTime-Strings wie '006:38:95' in Sekunden um.
    Erwartete Formate: 'MM:SS:CC' oder 'MM:SS'.
    """
    if gt is None or (isinstance(gt, float) and pd.isna(gt)):
        return None
    if isinstance(gt, (int, float)):
        return float(gt)
    gt = str(gt)
    parts = gt.split(":")
    try:
        if len(parts) == 3:
            mm, ss, cs = parts
            return int(mm) * 60 + int(ss) + int(cs) / 100.0
        elif len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + float(ss)
        else:
            return float(gt)
    except Exception:
        return None

# ============================================================
# Flanken mit Spielstand + roten Karten anreichern
# ============================================================

def enrich_crosses_with_state(match_id: str) -> pd.DataFrame:
    """
    Holt Flanken (03.06) und Basic-Events (03.05) für ein Match
    und fügt pro Flanke hinzu:
    - ScoreHome, ScoreGuest zum Flankenzeitpunkt
    - MatchStateForTeam: 'winning', 'drawing', 'losing'
    - AnyRedBeforeCross: ob vor der Flanke überhaupt eine rote Karte im Spiel war
    - RedForOwnTeam: rote Karte für das Team der Flanke vor diesem Zeitpunkt
    - RedForOppTeam: rote Karte für das gegnerische Team vor diesem Zeitpunkt

    Alle ursprünglichen Flanken-Attribute bleiben erhalten.
    """
    # Flanken aus 03.06
    crosses = get_crosses_for_match(match_id).copy()
    if crosses.empty:
        return crosses

    # Basic-Events 03.05
    basic = get_match_basic(match_id)
    goals_df, red_df, meta = build_score_and_redcard_timeline(basic)

    home_id = meta["HomeTeamId"]
    guest_id = meta["GuestTeamId"]

    # --------------------------------------------------------
    # 1) Zeitspalten vereinheitlichen (naive datetime64[ns])
    # --------------------------------------------------------
    ct_raw = pd.to_datetime(
        crosses["SyncedEventTime"],
        utc=True,
        errors="coerce"
    )
    crosses["CrossTime"] = ct_raw.dt.tz_localize(None)

    valid_mask = crosses["CrossTime"].notna()
    crosses_valid = crosses[valid_mask].copy()
    crosses_invalid = crosses[~valid_mask].copy()

    # Goals-Timeline vorbereiten
    if not goals_df.empty:
        goals_df = goals_df.sort_values("EventTime")
    else:
        goals_df = pd.DataFrame(columns=["EventTime", "ScoreHome", "ScoreGuest"])

    # --------------------------------------------------------
    # 2) As-of-Merge für gültige Zeiten
    # --------------------------------------------------------
    if not crosses_valid.empty and not goals_df.empty:
        crosses_valid = crosses_valid.sort_values("CrossTime")
        crosses_valid = pd.merge_asof(
            crosses_valid,
            goals_df,
            left_on="CrossTime",
            right_on="EventTime",
            direction="backward"
        )
    else:
        crosses_valid["ScoreHome"] = 0
        crosses_valid["ScoreGuest"] = 0

    # Fehlende Scores bei validen Flanken mit 0:0 füllen
    crosses_valid["ScoreHome"] = crosses_valid.get("ScoreHome", 0).fillna(0).astype(int)
    crosses_valid["ScoreGuest"] = crosses_valid.get("ScoreGuest", 0).fillna(0).astype(int)

    # Für invalide Flanken (ohne Zeit): Default 0:0
    crosses_invalid["ScoreHome"] = 0
    crosses_invalid["ScoreGuest"] = 0

    # Wieder zusammenführen
    crosses_all = pd.concat([crosses_valid, crosses_invalid], ignore_index=True)

    # --------------------------------------------------------
    # 3) MatchStateForTeam
    # --------------------------------------------------------

    def match_state_for_team(row):
        team = row["TeamId"]
        sh = row["ScoreHome"]
        sg = row["ScoreGuest"]
        if team == home_id:
            if sh > sg:
                return "winning"
            elif sh == sg:
                return "drawing"
            else:
                return "losing"
        elif team == guest_id:
            if sg > sh:
                return "winning"
            elif sg == sh:
                return "drawing"
            else:
                return "losing"
        else:
            return "unknown"

    crosses_all["MatchStateForTeam"] = crosses_all.apply(match_state_for_team, axis=1)

    # --------------------------------------------------------
    # 4) Rote Karten – Timeline
    # --------------------------------------------------------
    if not red_df.empty:
        red_df = red_df.sort_values("EventTime")
        first_red_overall = red_df["EventTime"].min()
        first_red_by_team = red_df.groupby("TeamId")["EventTime"].min().to_dict()
    else:
        first_red_overall = None
        first_red_by_team = {}

    def any_red_before(cross_time):
        if first_red_overall is None or pd.isna(cross_time):
            return False
        return cross_time >= first_red_overall

    def red_for_own_team(row):
        t = row["TeamId"]
        ct = row["CrossTime"]
        frt = first_red_by_team.get(t)
        if frt is None or pd.isna(ct):
            return False
        return ct >= frt

    def red_for_opp_team(row):
        t = row["TeamId"]
        ct = row["CrossTime"]

        if t == home_id:
            opp = guest_id
        elif t == guest_id:
            opp = home_id
        else:
            return False

        frt = first_red_by_team.get(opp)
        if frt is None or pd.isna(ct):
            return False
        return ct >= frt

    crosses_all["AnyRedBeforeCross"] = crosses_all["CrossTime"].apply(any_red_before)
    crosses_all["RedForOwnTeam"] = crosses_all.apply(red_for_own_team, axis=1)
    crosses_all["RedForOppTeam"] = crosses_all.apply(red_for_opp_team, axis=1)

    # CrossTime und EventTime kannst du bei Bedarf später droppen
    return crosses_all

# ============================================================
# Flanke → Schuss-Verknüpfung (mit xG + Kontext)
# ============================================================

def link_crosses_to_shots_for_match(match_id, time_window_seconds=8.0) -> pd.DataFrame:
    """
    Verlinkt jede Flanke mit dem nächstfolgenden Schuss desselben Teams
    innerhalb eines Zeitfensters (default: 8 Sekunden).
    Nutzt die bereits angereicherten Flanken (Score, MatchState, Red-Flags).
    Gibt pro Zeile Flanken-Attribute + Enrichment + Shot-Attribute zurück.
    """
    # Enriched Flanken
    crosses = enrich_crosses_with_state(match_id).copy()
    if crosses.empty:
        return pd.DataFrame()

    # Schüsse
    shots = get_shots_for_match(match_id).copy()
    if shots.empty:
        return pd.DataFrame()

    crosses["t_sec"] = crosses["GameTime"].apply(gametime_to_seconds)
    shots["t_sec"] = shots["GameTime"].apply(gametime_to_seconds)

    links = []
    for _, c in crosses.dropna(subset=["TeamId", "t_sec"]).iterrows():
        team = c["TeamId"]
        t0 = c["t_sec"]

        cand = shots[
            (shots["TeamId"] == team) &
            (shots["t_sec"] >= t0) &
            (shots["t_sec"] <= t0 + time_window_seconds)
        ]
        if cand.empty:
            continue

        s = cand.sort_values("t_sec").iloc[0]

        links.append({
            # Flanke – Originalattribute
            "EventId":         c["EventId"],
            "TeamId":          c["TeamId"],
            "MatchId":         c["MatchId"],
            "PlayerId":        c["PlayerId"],
            "GameTime":        c["GameTime"],
            "X":               c["X"],
            "Y":               c["Y"],
            "X_rec":           c["X_rec"],
            "Y_rec":           c["Y_rec"],
            "Evaluation":      c["Evaluation"],
            "IsCross":         c["IsCross"],
            "IsCorner":        c["IsCorner"],
            "InGameSection":   c["InGameSection"],
            "NumDefInBox":     c["NumDefInBox"],
            "NumAttInBox":     c["NumAttInBox"],
            # Enrichment
            "ScoreHome":       c["ScoreHome"],
            "ScoreGuest":      c["ScoreGuest"],
            "MatchStateForTeam": c["MatchStateForTeam"],
            "AnyRedBeforeCross": c["AnyRedBeforeCross"],
            "RedForOwnTeam":     c["RedForOwnTeam"],
            "RedForOppTeam":     c["RedForOppTeam"],
            # Shot
            "ShotEventId":     s["EventId"],
            "ShotPlayerId":    s["PlayerId"],
            "ShotGameTime":    s["GameTime"],
            "ShotTimeSec":     s["t_sec"],
            "ShotX":           s["X"],
            "ShotY":           s["Y"],
            "ShotxG":          s["xG"],
            "ShotResult":      s["ShotResult"],
            "ShotInGameSection": s["InGameSection"],
            "ShotNumDefInBox":   s["NumDefInBox"],
            "ShotNumAttInBox":   s["NumAttInBox"],
        })

    return pd.DataFrame(links)

# ============================================================
# Batch-Verarbeitung über viele Matches
# ============================================================

def build_datasets_for_matches(match_ids):
    """
    Läuft über eine Liste von MatchIds und baut zwei DataFrames:
    - alle Flanken (enriched)
    - alle Flanken mit anschließendem Schuss (inkl. xG & Kontext)
    """
    all_crosses = []
    all_links = []

    for mid in match_ids:
        print(f"Verarbeite Match {mid} …")
        try:
            crosses = enrich_crosses_with_state(mid)
            if not crosses.empty:
                crosses["SourceMatchId"] = mid
                all_crosses.append(crosses)

            links = link_crosses_to_shots_for_match(mid)
            if not links.empty:
                links["SourceMatchId"] = mid
                all_links.append(links)
        except Exception as e:
            print(f"Fehler bei Match {mid}: {e}")

    crosses_df = pd.concat(all_crosses, ignore_index=True) if all_crosses else pd.DataFrame()
    links_df = pd.concat(all_links, ignore_index=True) if all_links else pd.DataFrame()
    return crosses_df, links_df

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------
    # 1) Ein Test-Match für Kontrolle
    # --------------------------------------
    test_match = "DFL-MAT-J03YU7"

    print(f"=== Flanken für Match {test_match} (03.06 + Spielstand + Rot) ===")
    crosses_enriched = enrich_crosses_with_state(test_match)
    print("Anzahl Flanken:", len(crosses_enriched))
    if not crosses_enriched.empty:
        print(
            crosses_enriched[[
                "EventId", "TeamId", "PlayerId", "Evaluation",
                "IsPass", "IsCross", "IsCorner",
                "InGameSection", "ReceiverId", "ReceptionId",
                "GameTime", "X", "Y", "X_rec", "Y_rec",
                "NumDefInBox", "NumAttInBox",
                "ScoreHome", "ScoreGuest", "MatchStateForTeam",
                "AnyRedBeforeCross", "RedForOwnTeam", "RedForOppTeam",
                "MatchId"
            ]].head(20).to_string(index=False)
        )

    print(f"\n=== Schüsse (ShotAtGoal) für Match {test_match} (03.06) ===")
    shots = get_shots_for_match(test_match)
    print("Anzahl Schüsse:", len(shots))
    if not shots.empty:
        print(shots.head(10).to_string(index=False))

    print(f"\n=== Flanke → Schuss (mit xG + Spielstand + Rot) für Match {test_match} ===")
    links_df_test = link_crosses_to_shots_for_match(test_match, time_window_seconds=8.0)
    print("Anzahl Flanken mit anschließendem Schuss:", len(links_df_test))
    if not links_df_test.empty:
        print(
            links_df_test[[
                "EventId", "TeamId", "PlayerId",
                "GameTime", "X", "Y", "X_rec", "Y_rec",
                "Evaluation", "IsCross", "IsCorner", "InGameSection",
                "NumDefInBox", "NumAttInBox",
                "ScoreHome", "ScoreGuest", "MatchStateForTeam",
                "AnyRedBeforeCross", "RedForOwnTeam", "RedForOppTeam",
                "ShotEventId", "ShotPlayerId", "ShotGameTime",
                "ShotTimeSec", "ShotX", "ShotY", "ShotxG",
                "ShotResult", "ShotInGameSection",
                "ShotNumDefInBox", "ShotNumAttInBox"
            ]].to_string(index=False)
        )

    # --------------------------------------
    # 2) Batch über die letzten 5 Saisons
    # --------------------------------------
    print("\n=== Batch über letzte 5 Saisons (Bundesliga & 2. Bundesliga) ===")
    seasons, match_ids, _ = get_match_ids_last_5_seasons()  # <-- hier _ statt 2 Variablen
    print(f"Seasons: {seasons}")
    print(f"Anzahl Matches: {len(match_ids)}")

    crosses_all, links_all = build_datasets_for_matches(match_ids)

    print("\nGesamt-Flanken (enriched):", len(crosses_all))
    print("Gesamt Flanken mit anschließendem Schuss:", len(links_all))

    # Optional: Speichern als Parquet (Pfad nach Bedarf anpassen)
    # crosses_all.to_parquet("crosses_enriched_last5seasons.parquet", index=False)
    # links_all.to_parquet("crosses_with_shots_enriched_last5seasons.parquet", index=False)
    # print("Parquet-Dateien gespeichert.")
