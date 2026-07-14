import { useCallback, useRef, useState } from 'react';

const MAX_BARS = 14;

export function useRhythmTap() {
    const timestamps = useRef<number[]>([]);
    const [heights, setHeights] = useState<number[]>(Array(MAX_BARS).fill(25));
    const [count, setCount] = useState(0);

    const tap = useCallback(() => {
        const now = Date.now();
        timestamps.current.push(now);
        setCount(timestamps.current.length);

        setHeights(() => {
            const ts = timestamps.current;
            const gaps: number[] = [];
            for (let i = 1; i < ts.length; i++) gaps.push(ts[i] - ts[i - 1]);
            const recentGaps = gaps.slice(-MAX_BARS);
            if (recentGaps.length === 0) return Array(MAX_BARS).fill(25);
            const maxGap = Math.max(...recentGaps, 1);
            const bars = recentGaps.map((g) => Math.round(15 + (1 - g / maxGap) * 85));
            const padded = Array(MAX_BARS - bars.length).fill(25).concat(bars);
            return padded.slice(-MAX_BARS);
        });
    }, []);

    const reset = useCallback(() => {
        timestamps.current = [];
        setHeights(Array(MAX_BARS).fill(25));
        setCount(0);
    }, []);

    return { tap, reset, heights, count };
}
