import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
    RecordingPresets,
    requestRecordingPermissionsAsync,
    setAudioModeAsync,
    useAudioRecorder,
} from 'expo-audio';
import React, { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, Card, Chip, H1, HandText, Mono, Wave } from '../components/UI';
import { colors } from '../theme';
import { useRhythmTap } from '../hooks/useRhythmTap';
import { RootStackParamList } from '../navigation/types';
import { ApiError, submitAudioSample, submitTrainingSample } from '../api/client';

type Nav = NativeStackNavigationProp<RootStackParamList>;
type Rt = RouteProp<RootStackParamList, 'Training'>;

const MIN_TAPS = 4;
const MAX_RECORD_MS = 15_000;
const MIN_RECORD_MS = 3_000;

type TeachMode = 'rhythm' | 'audio';

type SubmitState =
    | { status: 'idle' | 'submitting' }
    | { status: 'success'; title: string; artist: string }
    | { status: 'error'; message: string };

export default function TrainingScreen() {
    const navigation = useNavigation<Nav>();
    const { params } = useRoute<Rt>();
    // A tap sequence or recording carried over from a failed identification —
    // "teach it what you just tapped/played" instead of starting over.
    const [providedTaps, setProvidedTaps] = useState<number[] | null>(
        params?.taps && params.taps.length > 0 ? params.taps : null
    );
    const [mode, setMode] = useState<TeachMode>(params?.audioUri ? 'audio' : 'rhythm');
    const [query, setQuery] = useState('');
    const { tap, reset, heights, count, getRelativeTaps } = useRhythmTap();
    const [state, setState] = useState<SubmitState>({ status: 'idle' });

    // audio mode
    const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
    const [recording, setRecording] = useState(false);
    const [recordedUri, setRecordedUri] = useState<string | null>(params?.audioUri ?? null);
    const [recordSeconds, setRecordSeconds] = useState(params?.audioUri ? MIN_RECORD_MS / 1000 : 0);
    const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

    useEffect(() => {
        return () => {
            if (tickRef.current) clearInterval(tickRef.current);
        };
    }, []);

    const startRecording = async () => {
        try {
            const perm = await requestRecordingPermissionsAsync();
            if (!perm.granted) {
                setState({ status: 'error', message: 'Microphone permission is needed to teach by audio.' });
                return;
            }
            await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
            // prepare is required before every recording session, including redos
            await recorder.prepareToRecordAsync();
            recorder.record();
            setRecordedUri(null);
            setRecordSeconds(0);
            setRecording(true);
            tickRef.current = setInterval(() => {
                setRecordSeconds((s) => {
                    if ((s + 1) * 1000 >= MAX_RECORD_MS) {
                        stopRecording();
                    }
                    return s + 1;
                });
            }, 1000);
        } catch {
            setState({ status: 'error', message: "Couldn't start recording." });
        }
    };

    const stopRecording = async () => {
        if (tickRef.current) {
            clearInterval(tickRef.current);
            tickRef.current = null;
        }
        setRecording(false);
        try {
            await recorder.stop();
            setRecordedUri(recorder.uri);
        } catch {
            setState({ status: 'error', message: 'Recording failed — try again.' });
        }
    };

    const canSubmit =
        query.trim().length > 0 &&
        state.status !== 'submitting' &&
        (mode === 'rhythm'
            ? providedTaps !== null || count >= MIN_TAPS
            : recordedUri !== null && !recording && recordSeconds >= MIN_RECORD_MS / 1000);

    const submit = async () => {
        setState({ status: 'submitting' });
        try {
            const result =
                mode === 'rhythm'
                    ? await submitTrainingSample(query.trim(), providedTaps ?? getRelativeTaps())
                    : await submitAudioSample(query.trim(), recordedUri!);
            if (result.status === 'resolved' && result.song) {
                setState({ status: 'success', title: result.song.title, artist: result.song.artist });
            } else {
                setState({ status: 'error', message: result.message ?? "Couldn't find that song." });
                setQuery('');
            }
        } catch (err) {
            setState({
                status: 'error',
                message: err instanceof ApiError ? err.message : 'Something went wrong.',
            });
        }
    };

    const teachAnother = () => {
        reset();
        setQuery('');
        setProvidedTaps(null);
        setRecordedUri(null);
        setRecordSeconds(0);
        setState({ status: 'idle' });
    };

    if (state.status === 'success') {
        return (
            <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
                <View style={styles.successBlock}>
                    <Mono>learned!</Mono>
                    <H1 style={{ marginTop: 8, textAlign: 'center' }}>{state.title}</H1>
                    <HandText style={styles.successArtist} bold={false}>{state.artist}</HandText>
                    <Mono style={{ marginTop: 16, textAlign: 'center' }}>
                        SongFinder will recognize this song a little better now.
                    </Mono>
                </View>
                <View style={styles.footer}>
                    <Btn label="Teach another" onPress={teachAnother} />
                    <Btn label="Done" ghost style={{ marginTop: 10 }} onPress={() => navigation.goBack()} />
                </View>
            </SafeAreaView>
        );
    }

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            <View style={styles.header}>
                <Mono>help it learn</Mono>
                <H1 style={{ marginTop: 4 }}>Teach a song</H1>
            </View>

            <View style={styles.modeRow}>
                <Chip label="Tap the rhythm" on={mode === 'rhythm'} onPress={() => setMode('rhythm')} />
                <Chip label="Play the audio" on={mode === 'audio'} onPress={() => setMode('audio')} />
            </View>

            <Card style={styles.searchCard}>
                <View style={styles.searchIcon}>
                    <Mono>?</Mono>
                </View>
                <TextInput
                    value={query}
                    onChangeText={setQuery}
                    placeholder="Song title or artist"
                    placeholderTextColor="rgba(0,0,0,0.35)"
                    style={styles.input}
                    editable={state.status !== 'submitting'}
                />
            </Card>

            {state.status === 'error' && (
                <Mono color="#b3261e" style={{ marginTop: 8 }}>
                    {state.message}
                </Mono>
            )}

            {mode === 'rhythm' ? (
                providedTaps ? (
                    <>
                        <View style={styles.captionBlock}>
                            <HandText style={styles.caption} bold={false}>
                                Using the rhythm you just tapped ({providedTaps.length} taps)
                            </HandText>
                        </View>
                        <Pressable onPress={() => setProvidedTaps(null)} style={styles.pad}>
                            <HandText style={styles.padLabel}>↺</HandText>
                            <Mono style={{ marginTop: 4 }}>tap a fresh one instead</Mono>
                        </Pressable>
                    </>
                ) : (
                    <>
                        <View style={styles.captionBlock}>
                            <HandText style={styles.caption} bold={false}>Now tap how it goes</HandText>
                        </View>

                        <Pressable onPress={tap} style={styles.pad}>
                            <HandText style={styles.padLabel}>TAP</HandText>
                            <Mono style={{ marginTop: 4 }}>the melody</Mono>
                        </Pressable>

                        <Wave heights={heights} height={44} />
                    </>
                )
            ) : (
                <>
                    <View style={styles.captionBlock}>
                        <HandText style={styles.caption} bold={false}>
                            {recording
                                ? 'Recording — keep the music playing'
                                : recordedUri
                                ? 'Got it! Submit when ready'
                                : 'Now play the song out loud'}
                        </HandText>
                    </View>

                    <Pressable
                        onPress={recording ? stopRecording : startRecording}
                        style={[styles.pad, recording && styles.padRecording]}
                    >
                        <HandText style={styles.padLabel}>{recording ? 'STOP' : 'REC'}</HandText>
                        <Mono style={{ marginTop: 4 }}>
                            {recording
                                ? `${recordSeconds}s / ${MAX_RECORD_MS / 1000}s`
                                : recordedUri
                                ? `${recordSeconds}s recorded · tap to redo`
                                : `${MIN_RECORD_MS / 1000}–${MAX_RECORD_MS / 1000}s of the song`}
                        </Mono>
                    </Pressable>
                </>
            )}

            <View style={styles.footer}>
                <Btn
                    label={state.status === 'submitting' ? 'Submitting…' : mode === 'rhythm' ? 'Submit rhythm' : 'Submit recording'}
                    disabled={!canSubmit}
                    onPress={submit}
                />
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 18 },
    header: { paddingTop: 10, paddingBottom: 12 },
    modeRow: { flexDirection: 'row', gap: 8, paddingBottom: 12 },
    searchCard: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 12 },
    searchIcon: { width: 26, height: 26, borderRadius: 13, borderWidth: 2, borderColor: colors.ink, alignItems: 'center', justifyContent: 'center' },
    input: { flex: 1, fontSize: 15, color: colors.ink },
    captionBlock: { alignItems: 'center', paddingVertical: 18 },
    caption: { fontSize: 18, color: 'rgba(0,0,0,0.55)', textAlign: 'center' },
    pad: {
        alignSelf: 'center',
        width: 170,
        height: 170,
        borderRadius: 85,
        borderWidth: 3,
        borderStyle: 'dashed',
        borderColor: colors.ink,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 20,
    },
    padRecording: { borderStyle: 'solid', borderColor: colors.accent, backgroundColor: 'rgba(42,120,214,0.08)' },
    padLabel: { fontSize: 22 },
    footer: { paddingVertical: 16 },
    successBlock: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 10 },
    successArtist: { fontSize: 20, color: 'rgba(0,0,0,0.55)', marginTop: 4 },
});
