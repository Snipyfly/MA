import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from b_ma_datenimport import CUSTOMER_ID, TOKEN, http_get_xml


FEED_MATCH_INFO = "DFL-02.01-Match-Information"
DEFAULT_CROSSES_GLOB = "cleaned/flanken_*_openplay.csv"
DEFAULT_OUT_FORMATIONS = "cleaned/n_formationen_0201_match_team.csv"
DEFAULT_OUT_CROSSES = "cleaned/n_flanken_mit_formation_0201.csv"


def to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_lineup_counts(lineup_text: Any) -> Optional[List[int]]:
    if lineup_text is None:
        return None
    text = str(lineup_text).strip()
    if not text:
        return None

    match = re.search(r"(\d+(?:\s*-\s*\d+)+)", text)
    if not match:
        return None

    normalized = re.sub(r"\s+", "", match.group(1))
    try:
        counts = [int(part) for part in normalized.split("-")]
    except ValueError:
        return None

    if not counts:
        return None

    # Falls GK mitkodiert ist (selten): 1-4-4-2 -> 4-4-2
    if sum(counts) == 11 and counts[0] == 1 and len(counts) > 1:
        counts = counts[1:]

    return counts


def format_lineup(counts: Optional[Sequence[int]]) -> Optional[str]:
    if not counts:
        return None
    return "-".join(str(v) for v in counts)


def get_match_information_root(parsed: Dict[str, Any]) -> Dict[str, Any]:
    pdr = parsed.get("PutDataRequest") if isinstance(parsed, dict) else None
    if isinstance(pdr, dict):
        root = pdr.get("MatchInformation")
        if isinstance(root, dict):
            return root

    if isinstance(parsed, dict) and "General" in parsed and "Teams" in parsed:
        return parsed

    raise ValueError("Kein 'MatchInformation'-Knoten im 02.01-Feed gefunden.")


def fetch_match_info(match_id: str, customer_id: str, feed: str, token: Optional[str]) -> Dict[str, Any]:
    parsed = http_get_xml(customer_id=customer_id, feed=feed, match_id=match_id, token=token)
    return get_match_information_root(parsed)


def extract_formations_for_match(match_info_root: Dict[str, Any], requested_match_id: str) -> List[Dict[str, Any]]:
    general = match_info_root.get("General") if isinstance(match_info_root, dict) else None
    general = general if isinstance(general, dict) else {}

    match_id = str(general.get("@MatchId") or requested_match_id)
    game_title = general.get("@MatchTitle")
    season = general.get("@Season")
    competition = general.get("@CompetitionName")

    teams_node = match_info_root.get("Teams") if isinstance(match_info_root, dict) else None
    teams_node = teams_node if isinstance(teams_node, dict) else {}

    rows: List[Dict[str, Any]] = []
    for team in to_list(teams_node.get("Team")):
        if not isinstance(team, dict):
            continue

        lineup_raw = team.get("@LineUp")
        lineup_counts = parse_lineup_counts(lineup_raw)
        lineup_normalized = format_lineup(lineup_counts)

        rows.append(
            {
                "MatchId": match_id,
                "GameTitle": game_title,
                "Season": season,
                "Competition": competition,
                "TeamId": team.get("@TeamId"),
                "TeamName": team.get("@TeamName"),
                "TeamRole": team.get("@Role"),
                "LineUpRaw": lineup_raw,
                "LineUp": lineup_normalized,
                "LineUpCounts": lineup_counts,
            }
        )

    return rows


def find_cross_files(crosses_glob: str, cwd: Path) -> List[Path]:
    files = sorted(cwd.glob(crosses_glob))
    return [p for p in files if p.is_file()]


def resolve_match_col(columns: Sequence[str]) -> Optional[str]:
    if "MatchId" in columns:
        return "MatchId"
    if "SourceMatchId" in columns:
        return "SourceMatchId"
    return None


def collect_match_ids_from_cross_files(cross_files: Sequence[Path]) -> List[str]:
    ids: set[str] = set()

    for path in cross_files:
        header = pd.read_csv(path, nrows=0)
        cols = list(header.columns)
        match_col = resolve_match_col(cols)
        if match_col is None:
            continue

        subset = pd.read_csv(path, usecols=lambda c: c == match_col)
        vals = subset[match_col].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        ids.update(vals.unique().tolist())

    return sorted(ids)


def fetch_formations_for_matches(
    match_ids: Sequence[str],
    customer_id: str,
    feed: str,
    token: Optional[str],
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    rows: List[Dict[str, Any]] = []
    errors: List[Tuple[str, str]] = []

    for match_id in match_ids:
        try:
            root = fetch_match_info(match_id=match_id, customer_id=customer_id, feed=feed, token=token)
            rows.extend(extract_formations_for_match(match_info_root=root, requested_match_id=match_id))
        except Exception as exc:
            errors.append((match_id, str(exc)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["MatchId", "TeamId"], keep="first")

    return df, errors


def load_and_concat_crosses(cross_files: Sequence[Path]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for path in cross_files:
        df = pd.read_csv(path)
        df["SourceFile"] = path.name
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def merge_crosses_with_formations(df_crosses: pd.DataFrame, df_formations: pd.DataFrame) -> pd.DataFrame:
    if df_crosses.empty:
        return df_crosses.copy()
    if df_formations.empty:
        merged = df_crosses.copy()
        merged["LineUp"] = pd.NA
        merged["LineUpRaw"] = pd.NA
        merged["TeamRole"] = pd.NA
        return merged

    cols = list(df_crosses.columns)
    match_col = resolve_match_col(cols)
    if match_col is None:
        raise ValueError("Flanken-Daten haben keine MatchId/SourceMatchId-Spalte.")
    if "TeamId" not in cols:
        raise ValueError("Flanken-Daten haben keine TeamId-Spalte.")

    left = df_crosses.copy()
    left["_match_id"] = left[match_col].astype(str)
    left["_team_id"] = left["TeamId"].astype(str)

    right = df_formations.copy()
    right["_match_id"] = right["MatchId"].astype(str)
    right["_team_id"] = right["TeamId"].astype(str)

    keep_cols = [
        "_match_id",
        "_team_id",
        "LineUp",
        "LineUpRaw",
        "TeamRole",
        "TeamName",
        "Season",
        "Competition",
        "GameTitle",
    ]

    merged = left.merge(right[keep_cols], on=["_match_id", "_team_id"], how="left")
    merged = merged.drop(columns=["_match_id", "_team_id"])
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Liest Formationen aus DFL-02.01 und ordnet sie Flanken (MatchId+TeamId) zu."
        )
    )
    parser.add_argument(
        "--match-id",
        action="append",
        default=[],
        help="MatchId (mehrfach nutzbar). Wenn leer, werden MatchIds aus Flanken-Dateien gezogen.",
    )
    parser.add_argument(
        "--crosses-glob",
        default=DEFAULT_CROSSES_GLOB,
        help=f"Glob fuer Flanken-Dateien relativ zum Projekt. Default: {DEFAULT_CROSSES_GLOB}",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Nur Formationen ziehen/speichern, ohne Flanken-Merge.",
    )
    parser.add_argument("--customer-id", default=CUSTOMER_ID, help="DFL Customer ID")
    parser.add_argument("--feed", default=FEED_MATCH_INFO, help="Feed Name (default 02.01)")
    parser.add_argument("--token", default=TOKEN, help="Bearer Token / DFL_API_TOKEN")
    parser.add_argument("--out-formations", default=DEFAULT_OUT_FORMATIONS, help="Output CSV Formationen")
    parser.add_argument("--out-crosses", default=DEFAULT_OUT_CROSSES, help="Output CSV Flanken+Formation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    cross_files = find_cross_files(args.crosses_glob, cwd=cwd)

    if args.match_id:
        match_ids = sorted(set(str(x).strip() for x in args.match_id if str(x).strip()))
    else:
        match_ids = collect_match_ids_from_cross_files(cross_files)

    if not match_ids:
        raise ValueError(
            "Keine MatchIds vorhanden. Gib --match-id an oder stelle sicher, dass Flanken-Dateien MatchId enthalten."
        )

    print(f"Anzahl MatchIds fuer 02.01-Abruf: {len(match_ids)}")

    df_formations, errors = fetch_formations_for_matches(
        match_ids=match_ids,
        customer_id=args.customer_id,
        feed=args.feed,
        token=args.token,
    )

    out_formations = (cwd / args.out_formations).resolve()
    out_formations.parent.mkdir(parents=True, exist_ok=True)
    df_formations.to_csv(out_formations, index=False)
    print(f"Formationen gespeichert: {out_formations}")
    print(f"Formations-Zeilen: {len(df_formations)}")

    if errors:
        print(f"Fehler bei {len(errors)} Matches:")
        for match_id, msg in errors[:20]:
            print(f"  {match_id}: {msg}")
        if len(errors) > 20:
            print(f"  ... und {len(errors)-20} weitere")

    if args.skip_merge:
        return

    if not cross_files:
        print("Keine Flanken-Dateien gefunden, Merge uebersprungen.")
        return

    print(f"Flanken-Dateien fuer Merge: {len(cross_files)}")
    df_crosses = load_and_concat_crosses(cross_files)
    merged = merge_crosses_with_formations(df_crosses=df_crosses, df_formations=df_formations)

    out_crosses = (cwd / args.out_crosses).resolve()
    out_crosses.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_crosses, index=False)
    print(f"Flanken+Formation gespeichert: {out_crosses}")
    print(f"Flanken-Zeilen (merged): {len(merged)}")


if __name__ == "__main__":
    main()
