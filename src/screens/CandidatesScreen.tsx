import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React, { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, Card, Chip, H2, HatchBox, Line, Mono } from '../components/UI';
import { colors } from '../theme';
import { MOCK_SONGS, Song } from '../data/songs';
import { RootStackParamList } from '../navigation/types';

type Nav = NativeStackNavigationProp<RootStackParamList>;
type Rt = RouteProp<RootStackParamList, 'Candidates'>;

const FILTERS = ['Genre', 'Decade', 'Tempo', 'Mood'] as const;

export default function CandidatesScreen() {
    const navigation = useNavigation<Nav>();
    const { params } = useRoute<Rt>();
    const [activeFilter, setActiveFilter] = useState<(typeof FILTERS)[number]>('Genre');

    const candidates: Song[] = useMemo(
        () =>
            MOCK_SONGS.filter((s) => s.id !== params?.excludeId).sort((a, b) => b.match - a.match),
        [params?.excludeId]
    );

    const openResult = (song: Song) => navigation.navigate('Result', { song, method: 'tapped' });

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            <View style={styles.header}>
                <H2 style={{ fontSize: 24 }}>{candidates.length} possible matches</H2>
                <Mono style={{ marginTop: 4 }}>narrow it down ↓</Mono>
            </View>

            <View style={styles.chipRow}>
                {FILTERS.map((f) => (
                    <Chip key={f} label={f} on={activeFilter === f} onPress={() => setActiveFilter(f)} />
                ))}
            </View>

            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ gap: 10, paddingBottom: 12 }}>
                {candidates.map((song) => (
                    <Pressable key={song.id} onPress={() => openResult(song)}>
                        <Card style={styles.row}>
                            <HatchBox width={44} height={44} radius={9} />
                            <View style={{ flex: 1, gap: 6 }}>
                                <Line width="70%" />
                                <Line width="45%" height={6} />
                            </View>
                            <Mono color={song.match >= 85 ? colors.accent : colors.mono}>{song.match}%</Mono>
                        </Card>
                    </Pressable>
                ))}
            </ScrollView>

            <View style={styles.footer}>
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
    footer: { paddingVertical: 14 },
});