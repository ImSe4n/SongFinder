"""How good is the auto-tap heuristic (app/audio/onsets.py), really?

For every song with at least one REAL human rhythm sample (source 'train' or
'confirm') AND at least one machine-extracted 'auto' sample, this compares
them via the same DTW distance the matcher uses — real ground truth vs. the
heuristic's guess for the same song.

This is honest ground-truth signal, not a full per-onset training set: we
know a human's *overall* tap pattern for the song, but not which exact audio
moment each of their taps corresponds to (that needs taps captured live
against playback, which the app doesn't do yet). So the result here is
song-level ("was the heuristic close for this song?"), which is exactly
enough to: (a) tell you honestly whether experiment #2 (auto-rhythm) is
working, and (b) serve as the objective function for tuning onsets.py's
constants against real examples instead of guessing.

Usage (from the backend/ directory):
    .venv/Scripts/python scripts/eval_auto_rhythm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import RhythmSample, Song
    from app.rhythm.matching import confidence_from_distance, dtw_distance

    with SessionLocal() as db:
        songs = db.scalars(select(Song)).all()

        results = []
        for song in songs:
            human = [s for s in song.samples if s.source in ("train", "confirm")]
            auto = [s for s in song.samples if s.source == "auto"]
            if not human or not auto:
                continue

            # Best-case pairing: closest auto sample to any human sample.
            best_distance = min(
                dtw_distance(h.log_ratio_vector, a.log_ratio_vector)
                for h in human
                for a in auto
            )
            results.append((song, best_distance, confidence_from_distance(best_distance)))

        if not results:
            print(
                "No songs have both a human rhythm sample and an auto sample "
                "yet — nothing to evaluate. Teach a song's audio (which seeds "
                "an auto sample) AND its rhythm by hand for at least one song "
                "to get a data point here."
            )
            return 0

        results.sort(key=lambda r: r[1])  # best (lowest distance) first

        print(f"{len(results)} song(s) with both a human and an auto rhythm sample:\n")
        for song, distance, confidence in results:
            verdict = "close" if confidence >= 70 else "off" if confidence >= 40 else "way off"
            print(f"  [{verdict:>7}] {confidence:5.1f}% (dist {distance:.3f})  {song.title} — {song.artist}")

        confidences = [c for _, _, c in results]
        mean_conf = sum(confidences) / len(confidences)
        median_conf = sorted(confidences)[len(confidences) // 2]
        close_count = sum(1 for c in confidences if c >= 70)

        print(
            f"\nSummary: mean {mean_conf:.1f}%, median {median_conf:.1f}%, "
            f"{close_count}/{len(results)} judged 'close' (>=70%).\n"
            f"This is the real accuracy of the onset-detection heuristic against "
            f"human ground truth — use it to judge whether AUTO_RHYTHM_ENABLED "
            f"and AUTO_RHYTHM_DISTANCE_PENALTY (backend/.env) are set sensibly, "
            f"and re-run after tweaking app/audio/onsets.py's constants."
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
