import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React from 'react';
import { Alert, Pressable, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, Card, H1, HandText, HatchBox, Line, Mono } from '../components/UI';
import { colors } from '../theme';
import { useLibrary } from '../store/LibraryContext';
import { RootStackParamList } from '../navigation/types';

type Nav = NativeStackNavigationProp<RootStackParamList>;
type Rt = RouteProp<RootStackParamList, 'Result'>;

export default function ResultScreen() {
    const navigation = useNavigation<Nav>();
    const { params } = useRoute<Rt>();
    const { song, method } = params;
    const { addEntry } = useLibrary();

    const save = () => {
        addEntry(song, method);
        navigation.navigate('MainTabs', { screen: 'History' } as any);
    };

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            <View style={styles.matchRow}>
                <View style={styles.matchChip}>
                    <HandText style={styles.matchLabel}>✓ {song.match}% match</HandText>
                </View>
            </View>

            <View style={styles.hero}>
                <HatchBox width={190} height={190} radius={18}>
                    <Mono>album art</Mono>
                </HatchBox>
                <View style={{ alignItems: 'center' }}>
                    <H1 style={styles.title}>{song.title}</H1>
                    <HandText style={styles.artist} bold={false}>{song.artist}</HandText>
                </View>
            </View>

            <View style={styles.links}>
                <Pressable onPress={() => Alert.alert('Lyrics', 'Lyrics view coming soon.')}>
                    <Card style={styles.linkCard}>
                        <View style={styles.linkIconSquare} />
                        <Line width={undefined as any} style={{ flex: 1 }} />
                        <Mono>lyrics</Mono>
                    </Card>
                </Pressable>
                <Pressable onPress={() => Alert.alert('Video', 'Video preview coming soon.')}>
                    <Card style={styles.linkCard}>
                        <View style={styles.linkIconCircle} />
                        <Line width={undefined as any} style={{ flex: 1 }} />
                        <Mono>video</Mono>
                    </Card>
                </Pressable>
            </View>

            <View style={{ flex: 1 }} />

            <View style={styles.footer}>
                <Btn label="Not it" ghost style={{ flex: 1 }} onPress={() => navigation.navigate('Candidates', { excludeId: song.id })} />
                <Btn label="Save to library" style={{ flex: 2 }} onPress={save} />
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 18 },
    matchRow: { alignItems: 'center', paddingTop: 10 },
    matchChip: { backgroundColor: colors.accent, borderRadius: 999, paddingVertical: 6, paddingHorizontal: 14 },
    matchLabel: { color: '#fff', fontSize: 14 },
    hero: { alignItems: 'center', gap: 14, paddingTop: 18 },
    title: { fontSize: 30, textAlign: 'center' },
    artist: { fontSize: 20, color: 'rgba(0,0,0,0.55)', marginTop: 2 },
    links: { gap: 10, paddingTop: 22 },
    linkCard: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 13, paddingHorizontal: 15 },
    linkIconSquare: { width: 22, height: 22, borderRadius: 5, borderWidth: 2, borderColor: colors.ink },
    linkIconCircle: { width: 22, height: 22, borderRadius: 11, borderWidth: 2, borderColor: colors.ink },
    footer: { flexDirection: 'row', gap: 10, paddingBottom: 18 },
});
