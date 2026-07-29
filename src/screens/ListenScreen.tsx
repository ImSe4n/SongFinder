import { CompositeNavigationProp, useNavigation } from '@react-navigation/native';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
    RecordingPresets,
    requestRecordingPermissionsAsync,
    setAudioModeAsync,
    useAudioRecorder,
} from 'expo-audio';
import React, { useEffect, useRef, useState } from 'react';
import { Animated, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, HandText, Line, Mono } from '../components/UI';
import { colors } from '../theme';
import { ApiError, queryAudio } from '../api/client';
import { MainTabParamList, RootStackParamList } from '../navigation/types';

type Nav = CompositeNavigationProp<
    BottomTabNavigationProp<MainTabParamList, 'Home'>,
    NativeStackNavigationProp<RootStackParamList>
>;

// ~8s of audio gives the fingerprinter plenty of hashes to align.
const LISTEN_MS = 8000;

type Phase = 'requesting' | 'listening' | 'matching' | 'denied' | 'no_match' | 'error';

const CAPTION: Record<Phase, string> = {
    requesting: 'Getting the mic ready…',
    listening: 'Play the song out loud',
    matching: 'Hmm, let me think…',
    denied: "Couldn't access the mic",
    no_match: "I don't know that one yet",
    error: 'Something went wrong',
};

const STATUS: Record<Phase, string> = {
    requesting: 'starting…',
    listening: 'listening…',
    matching: 'matching…',
    denied: 'microphone unavailable',
    no_match: 'no match found',
    error: 'connection trouble',
};

export default function ListenScreen() {
    const navigation = useNavigation<Nav>();
    const [phase, setPhase] = useState<Phase>('requesting');
    const [errorDetail, setErrorDetail] = useState<string | null>(null);
    const [attempt, setAttempt] = useState(0);
    const pulse = useRef(new Animated.Value(0)).current;
    const clipUriRef = useRef<string | null>(null);
    const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);

    useEffect(() => {
        let cancelled = false;
        let timer: ReturnType<typeof setTimeout>;

        (async () => {
            try {
                setPhase('requesting');
                setErrorDetail(null);
                clipUriRef.current = null;
                const perm = await requestRecordingPermissionsAsync();
                if (!perm.granted) {
                    if (!cancelled) setPhase('denied');
                    return;
                }
                await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
                await recorder.prepareToRecordAsync();
                if (cancelled) return;
                recorder.record();
                setPhase('listening');

                timer = setTimeout(async () => {
                    try {
                        await recorder.stop();
                        const uri = recorder.uri;
                        if (cancelled || !uri) return;
                        clipUriRef.current = uri;

                        setPhase('matching');
                        const result = await queryAudio(uri);
                        if (cancelled) return;

                        if (result.status === 'confident' && result.match) {
                            navigation.replace('Result', {
                                song: result.match,
                                method: 'listened',
                                candidates: result.candidates,
                                audioUri: uri,
                                queryLogId: result.queryLogId ?? undefined,
                            });
                        } else if (result.candidates.length > 0) {
                            navigation.replace('Candidates', {
                                candidates: result.candidates,
                                audioUri: uri,
                                queryLogId: result.queryLogId ?? undefined,
                            });
                        } else {
                            setPhase('no_match');
                        }
                    } catch (err) {
                        if (cancelled) return;
                        console.error('[ListenScreen] queryAudio failed:', err);
                        if (err instanceof ApiError) {
                            setErrorDetail(err.message);
                            setPhase('error');
                        } else {
                            setPhase('no_match');
                        }
                    }
                }, LISTEN_MS);
            } catch {
                if (!cancelled) setPhase('denied');
            }
        })();

        return () => {
            cancelled = true;
            clearTimeout(timer);
            recorder.stop().catch(() => {});
        };
    }, [attempt]);

    useEffect(() => {
        if (phase !== 'listening' && phase !== 'matching') return;
        const loop = Animated.loop(
            Animated.sequence([
                Animated.timing(pulse, { toValue: 1, duration: 900, useNativeDriver: true }),
                Animated.timing(pulse, { toValue: 0, duration: 900, useNativeDriver: true }),
            ])
        );
        loop.start();
        return () => loop.stop();
    }, [phase]);

    const ringScale = pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.15] });
    const failed = phase === 'denied' || phase === 'no_match' || phase === 'error';

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            <View style={styles.top}>
                <Mono color="rgba(255,255,255,0.55)">{STATUS[phase]}</Mono>
            </View>

            <View style={styles.center}>
                <Animated.View style={[styles.ringOuter, { transform: [{ scale: ringScale }] }]} />
                <View style={styles.ringInner} />
                <View style={[styles.dot, failed && styles.dotFailed]}>
                    <View style={styles.dotIcon} />
                </View>
            </View>

            <View style={styles.textBlock}>
                <HandText style={styles.caption}>{CAPTION[phase]}</HandText>
                {phase === 'error' && errorDetail && (
                    <Mono color="rgba(255,255,255,0.5)" style={{ marginTop: 10, textAlign: 'center' }}>
                        {errorDetail}
                    </Mono>
                )}
                {phase === 'no_match' && (
                    <Mono color="rgba(255,255,255,0.5)" style={{ marginTop: 10, textAlign: 'center' }}>
                        teach it this recording so it knows next time
                    </Mono>
                )}
                {(phase === 'listening' || phase === 'matching') && (
                    <Line width="60%" style={{ marginTop: 10, alignSelf: 'center', backgroundColor: 'rgba(255,255,255,0.22)' }} />
                )}
            </View>

            <View style={{ flex: 0.5 }} />

            <View style={styles.footer}>
                {failed ? (
                    <>
                        {phase === 'no_match' && clipUriRef.current && (
                            <Btn
                                label="Teach this recording"
                                ghost
                                dark
                                style={{ width: 220, backgroundColor: colors.accent, borderColor: colors.accent }}
                                onPress={() => navigation.navigate('Training', { audioUri: clipUriRef.current! })}
                            />
                        )}
                        {phase !== 'denied' && (
                            <Btn label="Listen again" ghost dark style={{ width: 220 }} onPress={() => setAttempt((a) => a + 1)} />
                        )}
                        <Btn label="Tap the rhythm instead" ghost dark style={{ width: 220 }} onPress={() => navigation.navigate('Tap')} />
                    </>
                ) : (
                    <>
                        <Mono color="rgba(255,255,255,0.5)">no music playing?</Mono>
                        <Btn label="Tap the rhythm instead" ghost dark style={{ width: 220 }} onPress={() => navigation.navigate('Tap')} />
                    </>
                )}
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.ink, paddingHorizontal: 20 },
    top: { alignItems: 'center', paddingTop: 8 },
    center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
    ringOuter: { position: 'absolute', width: 260, height: 260, borderRadius: 130, borderWidth: 1.5, borderColor: 'rgba(255,255,255,0.15)' },
    ringInner: { position: 'absolute', width: 200, height: 200, borderRadius: 100, borderWidth: 1.5, borderColor: 'rgba(255,255,255,0.28)' },
    dot: {
        width: 140,
        height: 140,
        borderRadius: 70,
        backgroundColor: colors.accent,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: colors.accent,
        shadowOpacity: 0.6,
        shadowRadius: 30,
        shadowOffset: { width: 0, height: 0 },
    },
    dotFailed: { backgroundColor: 'rgba(255,255,255,0.18)', shadowOpacity: 0 },
    dotIcon: { width: 36, height: 54, borderRadius: 18, borderWidth: 3, borderColor: '#fff' },
    textBlock: { alignItems: 'center', paddingHorizontal: 10 },
    caption: { fontSize: 22, color: '#fff', textAlign: 'center' },
    footer: { alignItems: 'center', gap: 10, paddingBottom: 18 },
});
