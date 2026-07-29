import { Song } from '../data/songs';
import { FindMethod } from '../store/LibraryContext';

export type RootStackParamList = {
Onboarding: undefined;
MainTabs: undefined;
Listen: undefined;
Result: {
  song: Song;
  method: FindMethod;
  taps?: number[];
  candidates?: Song[];
  audioUri?: string;
  mode?: 'identify' | 'browse';
  queryLogId?: string;
};
Candidates: {
  candidates: Song[];
  excludeId?: string;
  taps?: number[];
  audioUri?: string;
  queryLogId?: string;
};
Training: { taps?: number[]; audioUri?: string } | undefined;
Find: undefined;
};

export type MainTabParamList = {
Home: undefined;
Tap: undefined;
History: undefined;
You: undefined;
};
