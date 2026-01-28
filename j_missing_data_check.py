# j_missing_data_check.py
from pathlib import Path

import pandas as pd

from i_diagnose_event_assignment import diagnose_assignment, load_cross_events, load_segments

CLEANED_DIR = Path(__file__).resolve().parent / "cleaned"
OUT_FP = CLEANED_DIR / "j_missing_data_summary.csv"


def build_missing_data_summary(ev2: pd.DataFrame) -> pd.DataFrame:
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
    return df_sum


def main() -> None:
    seg = load_segments()
    ev = load_cross_events()

    ev2 = diagnose_assignment(ev, seg)

    # Fokus: nur fehlende/inkonsistente Daten
    missing_mask = (
        ev2["time_na"]
        | ev2["outside_segment_range"]
        | ev2["not_assignable_despite_time"]
        | ev2["ambiguous_multi_segment"]
    )

    df_sum = build_missing_data_summary(ev2)
    print("\n=== CHECK: Fehlende/inkonsistente Daten (Open-Play-Flanken) ===")
    print(df_sum)

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

    OUT_FP.parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(OUT_FP, index=False)
    print(f"\nSummary gespeichert: {OUT_FP}")


if __name__ == "__main__":
    main()
