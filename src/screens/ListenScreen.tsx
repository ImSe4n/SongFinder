import { CompositeNavigationProp, useNavigation } from '@react-navigation/native';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Audio } from 'expo-av';
import React, { useEffect, useRef, useState } from 'react';
import { Animated, Pressable, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, HandText, Line, Mono } from '../components/UI';
import { colors } from '../theme';
import { randomMatch } from '../data/songs';
import { MainTabParamList, RootStackParamList } from '../navigation/types';

type Nav = CompositeNavigationProp<
    BottomTabNavigationProp<MainTabParamList, 'Home'>,
    NativeStackNavigationProp<RootStackParamList>
>;

const LISTEN_MS = 3200;

export default function ListenScreen() {
    const navigation = useNavigation<Nav>();
    const [status, setStatus] = useState<'requesting' | 'listening' | 'denied'>('requesting');
    const pulse = useRef(new Animated.Value(0)).current;
    const recordingRef = useRef<Audio.Recording | null>(null);

    useEffect(() => {
        let cancelled = false;
        let timer: ReturnType<typeof setTimeout>;

        (async () => {
            try {
                const perm = await Audio.requestPermissionsAsync();
                if (!perm.granted) {
                    if (!cancelled) setStatus('denied');
                    return;
                }
                await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
                const { recording } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
                recordingRef.current = recording;
                if (cancelled) return;
                setStatus('listening');

                timer = setTimeout(async () => {
                    try {
                        await recordingRef.current?.stopAndUnloadAsync();
                    } catch { }
                    if (!cancelled) {
                        const song = randomMatch();
                        navigation.replace('Result', { song, method: 'listened' });
                    }
                }, LISTEN_MS);
            } catch {
                if (!cancelled) setStatus('denied');
            }
        })();

        return () => {
            cancelled = true;
            clearTimeout(timer);
            recordingRef.current?.stopAndUnloadAsync().catch(() => { });
        };
    }, []);

    useEffect(() => {
        if (status !== 'listening') return;
        const loop = Animated.loop(
            Animated.sequence([
                Animated.timing(pulse, { toValue: 1, duration: 900, useNativeDriver: true }),
                Animated.timing(pulse, { toValue: 0, duration: 900, useNativeDriver: true }),
            ])
        );
        loop.start();
        return () => loop.stop();
    }, [status]);

    const ringScale = pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.15] });

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            <View style={styles.top}>
                <Mono color="rgba(255,255,255,0.55)">
                    {status === 'denied' ? 'microphone unavailable' : 'listening…'}
                </Mono>
            </View>

            <View style={styles.center}>
                <Animated.View style={[styles.ringOuter, { transform: [{ scale: ringScale }] }]} />
                <View style={styles.ringInner} />
                <View style={styles.dot}>
                    <View style={styles.dotIcon} />
                </View>
            </View>

            <View style={styles.textBlock}>
                <HandText style={styles.caption}>
                    {status === 'denied' ? "Couldn't access the mic" : 'Play the song out loud'}
                </HandText>
                <Line width="60%" style={{ marginTop: 10, alignSelf: 'center', backgroundColor: 'rgba(255,255,255,0.22)' }} />
            </View>

            <View style={{ flex: 0.5 }} />

            <View style={styles.footer}>
                <Mono color="rgba(255,255,255,0.5)">no music playing?</Mono>
                <Btn
                    label="Tap the rhythm instead"
                    ghost
                    dark
                    style={{ width: 220 }}
                    onPress={() => navigation.navigate('Tap')}
                />
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
    dotIcon: { width: 36, height: 54, borderRadius: 18, borderWidth: 3, borderColor: '#fff' },
    textBlock: { alignItems: 'center', paddingHorizontal: 10 },
    caption: { fontSize: 22, color: '#fff' },
    footer: { alignItems: 'center', gap: 10, paddingBottom: 18 },
});