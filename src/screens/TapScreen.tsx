import { CompositeNavigationProp, useNavigation } from '@react-navigation/native';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, H1, HandText, Line, Mono, Wave } from '../components/UI';
import { colors } from '../theme';
import { useRhythmTap } from '../hooks/useRhythmTap';
import { randomMatch } from '../data/songs';
import { MainTabParamList, RootStackParamList } from '../navigation/types';

type Nav = CompositeNavigationProp<
    BottomTabNavigationProp<MainTabParamList, 'Tap'>,
    NativeStackNavigationProp<RootStackParamList>
>;

const MIN_TAPS = 6;

export default function TapScreen() {
    const navigation = useNavigation<Nav>();
    const { tap, reset, heights, count } = useRhythmTap();

    const findIt = () => {
        const song = randomMatch();
        navigation.navigate('Result', { song, method: 'tapped' });
        reset();
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
                    {count === 0 ? 'start tapping' : `${count} taps · keep going`}
                </Mono>
            </View>

            <View style={styles.footer}>
                <Pressable onPress={tap} style={({ pressed }) => [styles.pad, pressed && styles.padPressed]}>
                    <HandText style={styles.padLabel}>TAP</HandText>
                    <Mono style={{ marginTop: 4 }}>here</Mono>
                </Pressable>

                <View style={styles.actions}>
                    <Btn label="Reset" ghost style={{ flex: 1 }} onPress={reset} />
                    <Btn
                        label="Find it →"
                        style={{ flex: 2 }}
                        disabled={count < MIN_TAPS}
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
    padLabel: { fontSize: 26 },
    actions: { flexDirection: 'row', gap: 12, width: '100%' },
});
