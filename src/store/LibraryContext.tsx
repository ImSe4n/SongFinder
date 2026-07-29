import AsyncStorage from '@react-native-async-storage/async-storage';
import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { Song } from '../data/songs';

export type FindMethod = 'tapped' | 'listened';

export type LibraryEntry = {
  id: string;
  song: Song;
  method: FindMethod;
  savedAt: number;
};

type LibraryContextValue = {
  entries: LibraryEntry[];
  onboardingSeen: boolean;
  addEntry: (song: Song, method: FindMethod) => void;
  finishOnboarding: () => void;
  loaded: boolean;
};

const LibraryContext = createContext<LibraryContextValue | undefined>(undefined);

const ENTRIES_KEY = 'songfinder.library.entries';
const ONBOARDING_KEY = 'songfinder.onboarding.seen';

export function LibraryProvider({ children }: { children: React.ReactNode }) {
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [onboardingSeen, setOnboardingSeen] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [rawEntries, rawOnboarding] = await Promise.all([
          AsyncStorage.getItem(ENTRIES_KEY),
          AsyncStorage.getItem(ONBOARDING_KEY),
        ]);
        if (rawEntries) setEntries(JSON.parse(rawEntries));
        if (rawOnboarding === '1') setOnboardingSeen(true);
      } finally {
        setLoaded(true);
      }
    })();
  }, []);

  const addEntry: LibraryContextValue['addEntry'] = (song, method) => {
    setEntries((prev) => {
      const next = [{ id: `${Date.now()}`, song, method, savedAt: Date.now() }, ...prev];
      AsyncStorage.setItem(ENTRIES_KEY, JSON.stringify(next)).catch(() => {});
      return next;
    });
  };

  const finishOnboarding = () => {
    setOnboardingSeen(true);
    AsyncStorage.setItem(ONBOARDING_KEY, '1').catch(() => {});
  };

  const value = useMemo(
    () => ({ entries, onboardingSeen, addEntry, finishOnboarding, loaded }),
    [entries, onboardingSeen, loaded]
  );

  return <LibraryContext.Provider value={value}>{children}</LibraryContext.Provider>;
}

export function useLibrary() {
  const ctx = useContext(LibraryContext);
  if (!ctx) throw new Error('useLibrary must be used within LibraryProvider');
  return ctx;
}
