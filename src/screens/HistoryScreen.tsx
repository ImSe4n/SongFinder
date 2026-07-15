import { useNavigation } from '@react-navigation/native';
import type { CompositeNavigationProp } from '@react-navigation/native';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React, { useMemo, useState } from 'react';
import { Pressable, SectionList, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card, H1, HatchBox, Line, Mono, Chip } from '../components/UI';
import { colors } from '../theme';
import { useLibrary, LibraryEntry } from '../store/LibraryContext';
import { MainTabParamList, RootStackParamList } from '../navigation/types';

type Nav = CompositeNavigationProp<
    BottomTabNavigationProp<MainTabParamList, 'History'>,
    NativeStackNavigationProp<RootStackParamList>
>;

const FILTERS = ['All', 'Tapped', 'Listened'] as const;

function dayBucket(ts: number) {
    const d = new Date(ts);
    const now = new Date();
    const startOf = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
    const diffDays = Math.round((startOf(now) - startOf(d)) / 86400000);
    if (diffDays === 0) return 'today';
    if (diffDays === 1) return 'yesterday';
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export default function HistoryScreen() {
    const navigation = useNavigation<Nav>();
    const { entries } = useLibrary();
    const [filter, setFilter] = useState<(typeof FILTERS)[number]>('All');

    const filtered = useMemo(
        () => entries.filter((e) => (filter === 'All' ? true : e.method === (filter === 'Tapped' ? 'tapped' : 'listened'))),
        [entries, filter]
    );

    const thisMonthCount = useMemo(() => {
        const now = new Date();
        return entries.filter((e) => {
            const d = new Date(e.savedAt);
            return d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
        }).length;
    }, [entries]);

    const sections = useMemo(() => {
        const groups = new Map<string, LibraryEntry[]>();
        filtered.forEach((e) => {
            const key = dayBucket(e.savedAt);
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key)!.push(e);
        });
        return Array.from(groups.entries()).map(([title, data]) => ({ title, data }));
    }, [filtered]);

    return (
        <SafeAreaView style={styles.screen} edges={['top']}>
            <View style={styles.header}>
                <H1>Your finds</H1>
                <Mono style={{ marginTop: 4 }}>{thisMonthCount} songs · this month</Mono>
            </View>

            <View style={styles.chipRow}>
                {FILTERS.map((f) => (
                    <Chip key={f} label={f} on={filter === f} onPress={() => setFilter(f)} />
                ))}
            </View>

            {sections.length === 0 ? (
                <View style={styles.empty}>
                    <Mono>no finds yet — go tap or listen for a song</Mono>
                </View>
            ) : (
                <SectionList
                    sections={sections}
                    keyExtractor={(item) => item.id}
                    contentContainerStyle={{ paddingBottom: 20 }}
                    renderSectionHeader={({ section: { title } }) => <Mono style={styles.sectionLabel}>{title}</Mono>}
                    renderItem={({ item }) => (
                        <Pressable onPress={() => navigation.navigate('Result', { song: item.song, method: item.method })}>
                            <Card style={styles.row}>
                                <HatchBox width={40} height={40} radius={8} />
                                <View style={{ flex: 1, gap: 6 }}>
                                    <Line width="70%" />
                                    <Line width="42%" height={6} />
                                </View>
                                <Mono>▶</Mono>
                            </Card>
                        </Pressable>
                    )}
                    ItemSeparatorComponent={() => <View style={{ height: 9 }} />}
                />
            )}
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 18 },
    header: { paddingTop: 6, paddingBottom: 12 },
    chipRow: { flexDirection: 'row', gap: 6, paddingBottom: 12 },
    sectionLabel: { paddingVertical: 8 },
    row: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 9 },
    empty: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingBottom: 80 },
});