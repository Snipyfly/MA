# season_scanner_bl_bl2.py
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict

BASE          = "https://httpget.distribution.production.datahub-sts.de/DeliveryPlatform/REST/PullOnce"
CHANNEL       = "sgf-m9hk-4u2a-7i6w"
FEED_SCHEDULE = "DFL-01.06-BaseData-Schedule"

# Nur diese Wettbewerbe wollen wir (BL + 2. BL)
WANTED_COMPS = {
    "DFL-COM-000002": "Bundesliga",
}

#"DFL-COM-000001": "Bundesliga",
#"DFL-COM-000002": "2. Bundesliga",

# Season-ID-Kandidaten
SEASON_PREFIX = "DFL-SEA-0001"
CAND_RANGES = [
    "K8",
    # ggf. weitere ergänzen (J0–JZ etc.), wenn du ältere Seasons brauchst
]

#"K5", "K6", "K7", "K8", "K9",

def pull_xml(url: str) -> ET.Element | None:
    try:
        r = requests.get(url, headers={"Accept": "application/xml"}, timeout=60)
        r.raise_for_status()
        return ET.fromstring(r.content)
    except Exception:
        return None


def parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    s = s.split("+")[0].replace("T", " ")
    if "." in s:
        s = s.split(".")[0]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def extract_match_id_from_fixture(fx: ET.Element) -> str | None:
    # 1) klassische Attribute
    for key in ("MatchId", "MatchID", "FixtureId", "FixtureID"):
        mid = fx.get(key)
        if isinstance(mid, str) and mid.startswith("DFL-MAT-"):
            return mid
    # 2) irgendwelche anderen Attribute
    for v in fx.attrib.values():
        if isinstance(v, str) and v.startswith("DFL-MAT-"):
            return v
    # 3) Kind-Knoten
    for child in fx:
        text = (child.text or "").strip()
        if text.startswith("DFL-MAT-"):
            return text
        for v in child.attrib.values():
            if isinstance(v, str) and v.startswith("DFL-MAT-"):
                return v
    return None


def collect_seasons_and_matches():
    """
    Liefert eine Liste von Dicts mit:
      season, comp_id, comp_name, kickoff (datetime), match_id,
      home_id, home_name, guest_id, guest_name
    Nur Bundesliga & 2. Bundesliga.
    """
    acc = defaultdict(list)

    for suffix in CAND_RANGES:
        season_id = f"{SEASON_PREFIX}{suffix}"
        url = f"{BASE}/{CHANNEL}/{FEED_SCHEDULE}/{season_id}"
        root = pull_xml(url)
        if root is None:
            continue

        fixtures = root.findall(".//Fixtures/Fixture") or root.findall(".//Fixture")
        if not fixtures:
            continue

        for fx in fixtures:
            comp_id = fx.get("CompetitionId")
            if comp_id not in WANTED_COMPS:
                continue

            comp_name = fx.get("CompetitionName") or WANTED_COMPS.get(comp_id, comp_id)
            ko = fx.get("KickoffTime") or fx.get("PlannedKickoffTime")
            dt = parse_dt(ko)
            mid = extract_match_id_from_fixture(fx)
            if not mid:
                continue

            home_id   = fx.get("HomeTeamId")
            home_name = fx.get("HomeTeamName")
            guest_id   = fx.get("GuestTeamId")
            guest_name = fx.get("GuestTeamName")

            acc[season_id].append({
                "season":    season_id,
                "comp_id":   comp_id,
                "comp_name": comp_name,
                "kickoff":   dt,
                "match_id":  mid,
                "home_id":   home_id,
                "home_name": home_name,
                "guest_id":  guest_id,
                "guest_name": guest_name,
            })

    # flache Liste
    rows = []
    for season_id, matches in acc.items():
        rows.extend(matches)

    # sortieren
    rows.sort(key=lambda r: (r["kickoff"] or datetime.min, r["match_id"]))
    return rows


def get_match_ids_last_5_seasons():
    rows = collect_seasons_and_matches()

    # Season-Startdatum bestimmen (erstes Kickoff pro Season)
    season_start = {}
    for r in rows:
        sid = r["season"]
        dt = r["kickoff"]
        if dt is None:
            continue
        if sid not in season_start or dt < season_start[sid]:
            season_start[sid] = dt

    # Seasons nach Startdatum sortieren und die letzten 5 nehmen
    sorted_seasons = sorted(season_start.items(), key=lambda x: x[1])
    last_5_seasons = [sid for sid, _ in sorted_seasons[-5:]]

    print("Verwende folgende Seasons (letzte 5):", last_5_seasons)

    # Alle Matches aus diesen Seasons holen (BL + 2. BL)
    filtered = [r for r in rows if r["season"] in last_5_seasons]
    match_ids = sorted({r["match_id"] for r in filtered})

    # Metadaten pro MatchId: season, comp, kickoff, home, away …
    meta_by_mid = {r["match_id"]: r for r in filtered}

    return last_5_seasons, match_ids, meta_by_mid



def get_match_ids_last_5_seasons_for_team(team_id: str):
    """
    Gibt (last_5_seasons, rows_for_team) zurück, wobei rows_for_team
    alle Spiele der letzten 5 Seasons enthalten, an denen das Team beteiligt ist.
    Jeder Eintrag in rows_for_team ist ein Dict mit:
      season, comp_id, comp_name, kickoff, match_id, home_id, home_name, guest_id, guest_name
    """

    rows = collect_seasons_and_matches()

    # Season-Startdatum bestimmen (erstes Kickoff pro Season)
    season_start = {}
    for r in rows:
        sid = r["season"]
        dt = r["kickoff"]
        if dt is None:
            continue
        if sid not in season_start or dt < season_start[sid]:
            season_start[sid] = dt

    sorted_seasons = sorted(season_start.items(), key=lambda x: x[1])
    last_5_seasons = [sid for sid, _ in sorted_seasons[-5:]]
    print("Verwende folgende Seasons (letzte 5):", last_5_seasons)

    # Filter: nur Spiele mit gewünschtem Team (home oder guest)
    filtered = [
        r for r in rows
        if r["season"] in last_5_seasons
        and (r["home_id"] == team_id or r["guest_id"] == team_id)
    ]

    # zur Kontrolle
    print(f"Gefundene Spiele für Team {team_id} in den letzten 5 Seasons: {len(filtered)}")

    # nach Datum sortieren (zur Sicherheit)
    filtered.sort(key=lambda r: (r["kickoff"] or datetime.min, r["match_id"]))

    return last_5_seasons, filtered


if __name__ == "__main__":
    TEAM_ID_SGF = "DFL-CLU-00000W"

    seasons, rows_for_team = get_match_ids_last_5_seasons_for_team(TEAM_ID_SGF)
    print("\nSeasons:", seasons)
    print("Anzahl Spiele für SGF:", len(rows_for_team))
    print("MatchIds (Auszug):", ", ".join(r["match_id"] for r in rows_for_team[:20]), "...")
