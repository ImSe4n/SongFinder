import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React, { useMemo, useState } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, Card, Chip, H2, HandText, HatchBox, Mono } from '../components/UI';
import { colors } from '../theme';
import { Song } from '../data/songs';
import { RootStackParamList } from '../navigation/types';

type Nav = NativeStackNavigationProp<RootStackParamList>;
type Rt = RouteProp<RootStackParamList, 'Candidates'>;

// Mood is dropped — Spotify no longer provides usable mood data, so a chip
// for it would be non-functional.
const FILTERS = ['Genre', 'Decade', 'Tempo'] as const;
type FilterKey = (typeof FILTERS)[number];
const FILTER_TO_SONG_KEY: Record<FilterKey, keyof Song> = {
    Genre: 'genre',
    Decade: 'decade',
    Tempo: 'tempo',
};

export default function CandidatesScreen() {
    const navigation = useNavigation<Nav>();
    const { params } = useRoute<Rt>();
    const [activeFilter, setActiveFilter] = useState<FilterKey | null>(null);

    const allCandidates: Song[] = useMemo(
        () =>
            (params?.candidates ?? [])
                .filter((s) => s.id !== params?.excludeId)
                .sort((a, b) => (b.match ?? 0) - (a.match ?? 0)),
        [params?.candidates, params?.excludeId]
    );

    const topCandidate = allCandidates[0];

    const candidates = useMemo(() => {
        if (!activeFilter || !topCandidate) return allCandidates;
        const key = FILTER_TO_SONG_KEY[activeFilter];
        return allCandidates.filter((s) => s[key] === topCandidate[key]);
    }, [allCandidates, activeFilter, topCandidate]);

    const openResult = (song: Song) =>
        navigation.navigate('Result', {
            song,
            method: params?.audioUri ? 'listened' : 'tapped',
            candidates: allCandidates,
            taps: params?.taps,
            audioUri: params?.audioUri,
            // Picking a candidate is just viewing it, not confirming it's
            // correct — the real outcome is logged when they explicitly hit
            // Save/Not it on the Result screen, so carry the log id forward.
            queryLogId: params?.queryLogId,
        });

    const canTeach = (params?.taps?.length ?? 0) > 0 || !!params?.audioUri;

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            <View style={styles.header}>
                <H2 style={{ fontSize: 24 }}>{candidates.length} possible matches</H2>
                <Mono style={{ marginTop: 4 }}>narrow it down ↓</Mono>
            </View>

            <View style={styles.chipRow}>
                {FILTERS.map((f) => (
                    <Chip
                        key={f}
                        label={f}
                        on={activeFilter === f}
                        onPress={() => setActiveFilter((prev) => (prev === f ? null : f))}
                    />
                ))}
            </View>

            {candidates.length === 0 ? (
                <View style={styles.empty}>
                    <Mono>no candidates to show</Mono>
                </View>
            ) : (
                <ScrollView style={{ flex: 1 }} contentContainerStyle={{ gap: 10, paddingBottom: 12 }}>
                    {candidates.map((song) => (
                        <Pressable key={song.id} onPress={() => openResult(song)}>
                            <Card style={styles.row}>
                                {song.albumArtUrl ? (
                                    <Image source={{ uri: song.albumArtUrl }} style={styles.art} />
                                ) : (
                                    <HatchBox width={44} height={44} radius={9} />
                                )}
                                <View style={{ flex: 1, gap: 2 }}>
                                    <HandText style={styles.rowTitle} numberOfLines={1}>{song.title}</HandText>
                                    <Mono numberOfLines={1}>{song.artist}</Mono>
                                </View>
                                {song.match != null && (
                                    <Mono color={song.match >= 85 ? colors.accent : colors.mono}>{song.match}%</Mono>
                                )}
                            </Card>
                        </Pressable>
                    ))}
                </ScrollView>
            )}

            <View style={styles.footer}>
                {canTeach && (
                    <Btn
                        label="None of these — teach it"
                        onPress={() =>
                            navigation.navigate('Training', { taps: params?.taps, audioUri: params?.audioUri })
                        }
                    />
                )}
                <Btn label="Re-tap the rhythm" ghost onPress={() => navigation.navigate('MainTabs', { screen: 'Tap' } as any)} />
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 18 },
    header: { paddingTop: 10, paddingBottom: 10 },
    chipRow: { flexDirection: 'row', gap: 6, paddingBottom: 12, flexWrap: 'wrap' },
    row: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 10 },
    art: { width: 44, height: 44, borderRadius: 9 },
    rowTitle: { fontSize: 17 },
    footer: { paddingVertical: 14, gap: 10 },
    empty: { flex: 1, alignItems: 'center', justifyContent: 'center' },
});