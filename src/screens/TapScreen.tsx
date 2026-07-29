import { CompositeNavigationProp, useNavigation } from '@react-navigation/native';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React, { useState } from 'react';
import { Alert, Pressable, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, H1, HandText, Line, Mono, Wave } from '../components/UI';
import { colors } from '../theme';
import { useRhythmTap } from '../hooks/useRhythmTap';
import { ApiError, queryRhythm } from '../api/client';
import { MainTabParamList, RootStackParamList } from '../navigation/types';

type Nav = CompositeNavigationProp<
    BottomTabNavigationProp<MainTabParamList, 'Tap'>,
    NativeStackNavigationProp<RootStackParamList>
>;

const MIN_TAPS = 6;

export default function TapScreen() {
    const navigation = useNavigation<Nav>();
    const { tap, reset, heights, count, getRelativeTaps } = useRhythmTap();
    const [loading, setLoading] = useState(false);

    const findIt = async () => {
        const taps = getRelativeTaps();
        setLoading(true);
        try {
            const result = await queryRhythm(taps);

            if (result.status === 'confident' && result.match) {
                navigation.navigate('Result', {
                    song: result.match,
                    method: 'tapped',
                    taps,
                    candidates: result.candidates,
                    queryLogId: result.queryLogId ?? undefined,
                });
                reset();
            } else if (result.candidates.length > 0) {
                navigation.navigate('Candidates', {
                    candidates: result.candidates,
                    taps,
                    queryLogId: result.queryLogId ?? undefined,
                });
                reset();
            } else {
                Alert.alert(
                    "Don't know that one yet",
                    'SongFinder has never heard this rhythm. Want to teach it?',
                    [
                        { text: 'Not now', style: 'cancel' },
                        {
                            text: 'Teach this rhythm',
                            onPress: () => {
                                navigation.navigate('Training', { taps });
                                reset();
                            },
                        },
                    ]
                );
            }
        } catch (err) {
            const message = err instanceof ApiError ? err.message : 'Something went wrong.';
            Alert.alert('Could not find a match', message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            <View style={styles.header}>
                <Pressable onPress={() => navigation.navigate('Home')}>
                    <Mono>← back</Mono>
                </Pressable>
                <Mono>tap the rhythm</Mono>
                <View style={{ width: 44 }} />
            </View>

            <View style={styles.titleBlock}>
                <H1 style={styles.title}>Tap it like{'\n'}you'd sing it</H1>
                <Line width="70%" style={{ marginTop: 12, alignSelf: 'center' }} />
            </View>

            <View style={styles.waveBlock}>
                <Wave heights={heights} height={90} />
                <Mono style={{ textAlign: 'center', marginTop: 12 }}>
                    {loading ? 'finding…' : count === 0 ? 'start tapping' : `${count} taps · keep going`}
                </Mono>
            </View>

            <View style={styles.footer}>
                <Pressable
                    onPress={tap}
                    disabled={loading}
                    style={({ pressed }) => [styles.pad, pressed && styles.padPressed, loading && styles.padDisabled]}
                >
                    <HandText style={styles.padLabel}>TAP</HandText>
                    <Mono style={{ marginTop: 4 }}>here</Mono>
                </Pressable>

                <View style={styles.actions}>
                    <Btn label="Reset" ghost style={{ flex: 1 }} disabled={loading} onPress={reset} />
                    <Btn
                        label={loading ? 'Finding…' : 'Find it →'}
                        style={{ flex: 2 }}
                        disabled={count < MIN_TAPS || loading}
                        onPress={findIt}
                    />
                </View>
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 20 },
    header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 10 },
    titleBlock: { alignItems: 'center', paddingTop: 8 },
    title: { fontSize: 32, textAlign: 'center', lineHeight: 34 },
    waveBlock: { flex: 1, justifyContent: 'center' },
    footer: { alignItems: 'center', gap: 20, paddingBottom: 16 },
    pad: {
        width: 190,
        height: 190,
        borderRadius: 95,
        borderWidth: 3,
        borderColor: colors.ink,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: colors.card,
    },
    padPressed: { backgroundColor: colors.line },
    padDisabled: { opacity: 0.5 },
    padLabel: { fontSize: 26 },
    actions: { flexDirection: 'row', gap: 12, width: '100%' },
});
