"""Where SongFinder actually stands re: training a real (ML) rhythm-matching
model, as opposed to the heuristics currently in place.

Neither DTW-on-log-ratio-intervals (the rhythm matcher) nor onset-detection
(the auto-tap heuristic) is machine learning — both are fixed, hand-written
algorithms. Training an actual model needs real labeled examples: (query,
correct song) pairs that came from a human, not from another algorithm's
guess. This script reports how much of that actually exists.

Usage (from the backend/ directory):
    .venv/Scripts/python scripts/training_status.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Rough, conservative rule of thumb for a first metric-learning attempt on a
# sequence task like this: enough examples to see real diversity of songs
# AND repeated examples per song (so the model can learn "these are the same
# rhythm" rather than just memorizing individual sequences).
MIN_HUMAN_SAMPLES_TOTAL = 300
MIN_SONGS_WITH_2PLUS_SAMPLES = 30


def main() -> int:
    from sqlalchemy import func, select

    from app.database import SessionLocal
    from app.models import QueryLog, RhythmSample, Song

    with SessionLocal() as db:
        total_songs = db.scalar(select(func.count(Song.id)))

        app_native_samples = db.scalars(
            select(RhythmSample).where(RhythmSample.source.in_(["train", "confirm"]))
        ).all()
        external_samples = db.scalars(
            select(RhythmSample).where(RhythmSample.source == "external_qbte")
        ).all()
        human_samples = app_native_samples + external_samples

        by_song: dict[int, int] = {}
        by_song_native: dict[int, int] = {}
        for s in human_samples:
            by_song[s.song_id] = by_song.get(s.song_id, 0) + 1
        for s in app_native_samples:
            by_song_native[s.song_id] = by_song_native.get(s.song_id, 0) + 1
        songs_with_2plus = sum(1 for c in by_song.values() if c >= 2)
        songs_with_2plus_native = sum(1 for c in by_song_native.values() if c >= 2)

        auto_samples = db.scalar(
            select(func.count(RhythmSample.id)).where(RhythmSample.source == "auto")
        )

        total_queries = db.scalar(select(func.count(QueryLog.id)))
        rhythm_queries = db.scalar(
            select(func.count(QueryLog.id)).where(QueryLog.query_type == "rhythm")
        )
        audio_queries = db.scalar(
            select(func.count(QueryLog.id)).where(QueryLog.query_type == "audio")
        )
        with_feedback = db.scalar(
            select(func.count(QueryLog.id)).where(QueryLog.outcome.is_not(None))
        )
        saved_count = db.scalar(
            select(func.count(QueryLog.id)).where(QueryLog.outcome == "saved")
        )

        print("=== Rhythm-matching ML readiness ===\n")
        print(f"App-native human samples (source='train'/'confirm', real SongFinder users): {len(app_native_samples)}")
        print(f"  Distinct songs with >=1: {len(by_song_native)} / {total_songs}   With >=2: {songs_with_2plus_native}")
        print(
            f"External human samples (source='external_qbte', imported from QBT-Extended "
            f"research dataset — CC BY-NC-SA, non-commercial only): {len(external_samples)}"
        )
        print(f"Machine-guessed 'auto' rhythm samples (NOT usable as training labels): {auto_samples}")

        print(f"\nCombined human-labeled total: {len(human_samples)}")
        print(f"  Distinct songs with >=1 human sample: {len(by_song)}   With >=2: {songs_with_2plus}")
        print(
            f"Threshold for a first real attempt: "
            f"{MIN_HUMAN_SAMPLES_TOTAL} total human samples, "
            f"{MIN_SONGS_WITH_2PLUS_SAMPLES} songs with 2+ each"
        )
        pct_total = 100 * len(human_samples) / MIN_HUMAN_SAMPLES_TOTAL
        pct_diversity = 100 * songs_with_2plus / MIN_SONGS_WITH_2PLUS_SAMPLES
        print(f"  Progress: {min(pct_total, 100):.0f}% (volume), {min(pct_diversity, 100):.0f}% (diversity)")
        if len(human_samples) >= MIN_HUMAN_SAMPLES_TOTAL and songs_with_2plus >= MIN_SONGS_WITH_2PLUS_SAMPLES:
            print("  -> Worth prototyping a real model now.")
            if len(app_native_samples) < MIN_HUMAN_SAMPLES_TOTAL:
                print(
                    "     (Note: this is mostly/entirely the external NC-licensed dataset — "
                    "fine for prototyping, but re-check licensing before any commercial use.)"
                )
        else:
            print("  -> Not there yet. Every real Training/Confirm submission moves this forward.")

        print("\n=== Query outcome log (backend/app/models.py: QueryLog) ===\n")
        print(f"Total logged queries: {total_queries} ({rhythm_queries} rhythm, {audio_queries} audio)")
        print(f"  With user feedback: {with_feedback} ({saved_count} saved)")
        print(
            "  This log is what future threshold auto-calibration and any "
            "'was this match actually correct' model would train on."
        )

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
