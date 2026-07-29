import { File } from 'expo-file-system';
import { Song } from '../data/songs';

const API_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';
const REQUEST_TIMEOUT_MS = 10_000;

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function rawRequest<T>(path: string, init: RequestInit, timeoutMs: number): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError('Request timed out — is the backend running?');
    }
    console.error(`[api] fetch failed for ${path}:`, err);
    throw new ApiError('Could not reach the server. Check your connection and API URL.');
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body = await response.json();
      detail = body?.detail ? String(body.detail) : undefined;
    } catch {
      // response wasn't JSON — fall through to statusText
    }
    throw new ApiError(detail ?? response.statusText, response.status);
  }

  return (await response.json()) as T;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return rawRequest<T>(
    path,
    { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } },
    REQUEST_TIMEOUT_MS
  );
}

// Audio uploads: larger payload, slower fingerprinting — longer timeout, and
// no explicit Content-Type so fetch sets the multipart boundary itself.
const UPLOAD_TIMEOUT_MS = 30_000;

function audioFormData(fileUri: string, extra?: Record<string, string>): FormData {
  const form = new FormData();
  // expo-file-system's File implements Blob, which RN's FormData/fetch (SDK
  // 57+) expect for file parts — the older { uri, name, type } object literal
  // throws "Unsupported FormDataPart implementation" on this SDK.
  form.append('file', new File(fileUri), 'clip.m4a');
  for (const [key, value] of Object.entries(extra ?? {})) {
    form.append(key, value);
  }
  return form;
}

type SongOut = {
  id: string;
  title: string;
  artist: string;
  genre: string;
  decade: string;
  tempo: 'slow' | 'medium' | 'fast' | null;
  spotifyTrackId: string;
  albumArtUrl: string | null;
  sampleCount: number;
};

type CandidateSongOut = SongOut & { match: number };

function candidateToSong(c: CandidateSongOut): Song {
  return { ...c, match: c.match };
}

function resolvedSongToSong(s: SongOut): Song {
  // Training resolves a song by catalog lookup, not by matching — there's no
  // meaningful "match %", so leave it absent.
  return { ...s };
}

export type QueryStatus = 'confident' | 'ambiguous' | 'no_match';

export type QueryResult = {
  status: QueryStatus;
  match: Song | null;
  candidates: Song[];
  queryLogId: string | null;
};

export async function queryRhythm(tapsMs: number[]): Promise<QueryResult> {
  const body = await request<{
    status: QueryStatus;
    match: CandidateSongOut | null;
    candidates: CandidateSongOut[];
    queryLogId: string | null;
  }>('/rhythm/query', {
    method: 'POST',
    body: JSON.stringify({ tapsMs }),
  });
  return {
    status: body.status,
    match: body.match ? candidateToSong(body.match) : null,
    candidates: body.candidates.map(candidateToSong),
    queryLogId: body.queryLogId,
  };
}

export type TrainingResult = {
  status: 'resolved' | 'not_found';
  song: Song | null;
  message: string | null;
};

export async function submitTrainingSample(
  query: string,
  tapsMs: number[]
): Promise<TrainingResult> {
  const body = await request<{
    status: 'resolved' | 'not_found';
    song: SongOut | null;
    message: string | null;
  }>('/rhythm/samples', {
    method: 'POST',
    body: JSON.stringify({ query, tapsMs }),
  });
  return {
    status: body.status,
    song: body.song ? resolvedSongToSong(body.song) : null,
    message: body.message,
  };
}

export type ConfirmResult = {
  status: 'ok';
  sampleCount: number;
};

export async function confirmSample(songId: string, tapsMs: number[]): Promise<ConfirmResult> {
  return request<ConfirmResult>('/rhythm/confirm', {
    method: 'POST',
    body: JSON.stringify({ songId, tapsMs }),
  });
}

export async function queryAudio(fileUri: string): Promise<QueryResult> {
  const body = await rawRequest<{
    status: QueryStatus;
    match: CandidateSongOut | null;
    candidates: CandidateSongOut[];
    queryLogId: string | null;
  }>('/audio/query', { method: 'POST', body: audioFormData(fileUri) }, UPLOAD_TIMEOUT_MS);
  return {
    status: body.status,
    match: body.match ? candidateToSong(body.match) : null,
    candidates: body.candidates.map(candidateToSong),
    queryLogId: body.queryLogId,
  };
}

export async function submitFeedback(
  queryLogId: string,
  outcome: 'saved' | 'not_it',
  songId?: string
): Promise<void> {
  // Best-effort telemetry — callers should never block or alert on this
  // failing (see call sites in ResultScreen/CandidatesScreen).
  await request<{ status: 'ok' }>('/rhythm/feedback', {
    method: 'POST',
    body: JSON.stringify({ queryLogId, outcome, songId }),
  });
}

export async function submitAudioSample(query: string, fileUri: string): Promise<TrainingResult> {
  const body = await rawRequest<{
    status: 'resolved' | 'not_found';
    song: SongOut | null;
    message: string | null;
  }>(
    '/audio/samples',
    { method: 'POST', body: audioFormData(fileUri, { query }) },
    UPLOAD_TIMEOUT_MS
  );
  return {
    status: body.status,
    song: body.song ? resolvedSongToSong(body.song) : null,
    message: body.message,
  };
}

export type LyricsSearchHit = {
  title: string;
  artist: string;
  album: string | null;
  snippet: string | null;
  songId?: string | null; // set when the hit is a song in our own catalog
  albumArtUrl?: string | null;
};

export async function confirmAudio(songId: string, fileUri: string): Promise<ConfirmResult> {
  return rawRequest<ConfirmResult>(
    '/audio/confirm',
    { method: 'POST', body: audioFormData(fileUri, { song_id: songId }) },
    UPLOAD_TIMEOUT_MS
  );
}

export async function searchLyrics(q: string): Promise<LyricsSearchHit[]> {
  const body = await request<{ results: LyricsSearchHit[] }>(
    `/songs/lyrics-search?q=${encodeURIComponent(q)}`
  );
  return body.results;
}

export type LyricsResult = {
  title: string | null;
  artist: string | null;
  lyrics: string | null;
};

export async function getLyrics(title: string, artist: string): Promise<LyricsResult> {
  return request<LyricsResult>(
    `/songs/lyrics?title=${encodeURIComponent(title)}&artist=${encodeURIComponent(artist)}`
  );
}

export type PreviewResult = {
  previewUrl: string | null;
  artworkUrl: string | null;
};

export async function getPreview(title: string, artist: string): Promise<PreviewResult> {
  return request<PreviewResult>(
    `/songs/preview?title=${encodeURIComponent(title)}&artist=${encodeURIComponent(artist)}`
  );
}

export type FilterParams = {
  artist?: string;
  genre?: string;
  decade?: string;
  tempo?: string;
};

export async function filterSongs(params: FilterParams): Promise<Song[]> {
  const qs = Object.entries(params)
    .filter(([, v]) => v)
    .map(([k, v]) => `${k}=${encodeURIComponent(v!)}`)
    .join('&');
  const body = await request<{ results: SongOut[] }>(`/songs/filter${qs ? `?${qs}` : ''}`);
  return body.results.map((s) => ({ ...s }));
}

export type Facets = {
  genres: string[];
  decades: string[];
  tempos: string[];
  artists: string[];
};

export async function getFacets(): Promise<Facets> {
  return request<Facets>('/songs/facets');
}
