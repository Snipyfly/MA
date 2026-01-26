import pandas as pd
from pathlib import Path
import re

CLEANED = Path("cleaned")

# 1) Match-IDs aus OpenPlay-Flanken-Dateien ziehen
flank_files = sorted(CLEANED.glob("flanken_*_openplay.csv"))
if not flank_files:
    raise FileNotFoundError("Keine flanken_*_openplay.csv in cleaned gefunden.")

all_flank_mids = set()
for fp in flank_files:
    df = pd.read_csv(fp)
    # bei dir ist MatchId in den Flanken-Dateien vorhanden
    if "MatchId" not in df.columns:
        raise ValueError(f"{fp.name}: MatchId-Spalte fehlt.")
    all_flank_mids |= set(df["MatchId"].dropna().astype(str).unique())

print("Unique Matches in OpenPlay-Flanken:", len(all_flank_mids))

# 2) Match-IDs aus Segment-CSV ziehen
seg_fp = CLEANED / "zeitsegmente_match_level.csv"
df_seg = pd.read_csv(seg_fp)

# Spalte heißt bei dir sehr wahrscheinlich match_id (oder MatchId) – robust:
match_col = None
for c in ["match_id", "MatchId", "matchId"]:
    if c in df_seg.columns:
        match_col = c
        break
if match_col is None:
    raise ValueError(f"Keine MatchId-Spalte in {seg_fp.name} gefunden. Spalten: {list(df_seg.columns)}")

seg_mids = set(df_seg[match_col].dropna().astype(str).unique())
print("Unique Matches in Zeitsegmenten:", len(seg_mids))

# 3) Fehlende Matches
missing = sorted(all_flank_mids - seg_mids)
extra = sorted(seg_mids - all_flank_mids)

print("\nFehlende Matches (in Flanken, aber nicht in Segmenten):", len(missing))
print("Beispiel:", missing[:20])

print("\nZusätzliche Matches (in Segmenten, aber nicht in Flanken):", len(extra))
print("Beispiel:", extra[:20])

# 4) Optional: prüfen, ob jedes Match überhaupt ein drawing/even Segment hat
# (aus logischer Sicht muss es das geben, sobald Kickoff/Ende da ist)
required_cols = {"match_state_for_team", "man_adv_state_for_team"}
if required_cols.issubset(df_seg.columns):
    # Hier zählt: match_id kommt mindestens einmal als drawing/even vor
    has_drawing_even = (
        df_seg.assign(_mid=df_seg[match_col].astype(str))
              .groupby("_mid")
              .apply(lambda g: ((g["match_state_for_team"] == "drawing") &
                               (g["man_adv_state_for_team"] == "even")).any())
    )

    missing_drawing_even = sorted(has_drawing_even[~has_drawing_even].index.tolist())
    print("\nMatches OHNE drawing/even Segment:", len(missing_drawing_even))
    print("Beispiel:", missing_drawing_even[:20])
else:
    print("\nHinweis: match_state_for_team / man_adv_state_for_team fehlen in Segmentdatei – drawing/even Check übersprungen.")
