export type Song = {
  id: string;
  title: string;
  artist: string;
  genre: string;
  decade: string;
  tempo: 'slow' | 'medium' | 'fast' | null;
  match?: number;  // rhythm/audio match confidence % — absent for browsed/filtered results
  spotifyTrackId?: string;
  albumArtUrl?: string | null;
  sampleCount?: number;
};

export const MOCK_SONGS: Song[] = [
  { id: 's1', title: 'Golden Hour', artist: 'Reya Lane', genre: 'Pop', decade: '2020s', tempo: 'medium', match: 96 },
  { id: 's2', title: 'Static Bloom', artist: 'The Quiet Wires', genre: 'Indie', decade: '2010s', tempo: 'medium', match: 88 },
  { id: 's3', title: 'Neon Tide', artist: 'Marlow', genre: 'Synthpop', decade: '2020s', tempo: 'fast', match: 81 },
  { id: 's4', title: 'Low Light', artist: 'Cassian Row', genre: 'R&B', decade: '2010s', tempo: 'slow', match: 77 },
  { id: 's5', title: 'Paper Planes Again', artist: 'Junebird', genre: 'Folk', decade: '2000s', tempo: 'slow', match: 72 },
  { id: 's6', title: 'Overdrive', artist: 'Kilo North', genre: 'Electronic', decade: '2020s', tempo: 'fast', match: 65 },
];

export function randomMatch(): Song {
  const base = MOCK_SONGS[Math.floor(Math.random() * MOCK_SONGS.length)];
  return { ...base, match: Math.floor(85 + Math.random() * 14) };
}
