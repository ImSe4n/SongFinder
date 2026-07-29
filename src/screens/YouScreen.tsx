import React, { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card, H1, Mono } from '../components/UI';
import { colors } from '../theme';
import { useLibrary } from '../store/LibraryContext';

export default function YouScreen() {
    const { entries } = useLibrary();

    const stats = useMemo(() => {
        const tapped = entries.filter((e) => e.method === 'tapped').length;
        const listened = entries.filter((e) => e.method === 'listened').length;
        return { total: entries.length, tapped, listened };
    }, [entries]);

    return (
        <SafeAreaView style={styles.screen} edges={['top']}>
            <View style={styles.header}>
                <H1>You</H1>
            </View>

            <View style={styles.statsRow}>
                <Card style={styles.statCard}>
                    <Mono>total finds</Mono>
                    <H1 style={styles.statNum}>{stats.total}</H1>
                </Card>
                <Card style={styles.statCard}>
                    <Mono>tapped</Mono>
                    <H1 style={styles.statNum}>{stats.tapped}</H1>
                </Card>
                <Card style={styles.statCard}>
                    <Mono>listened</Mono>
                    <H1 style={styles.statNum}>{stats.listened}</H1>
                </Card>
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 18 },
    header: { paddingTop: 6, paddingBottom: 18 },
    statsRow: { flexDirection: 'row', gap: 10 },
    statCard: { flex: 1, alignItems: 'center', gap: 8, paddingVertical: 18 },
    statNum: { fontSize: 26 },
});
