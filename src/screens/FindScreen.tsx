import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React, { useEffect, useMemo, useState } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card, Chip, H1, H2, HandText, HatchBox, Mono } from '../components/UI';
import { colors } from '../theme';
import { Song } from '../data/songs';
import { RootStackParamList } from '../navigation/types';
import {
    Facets,
    LyricsSearchHit,
    filterSongs,
    getFacets,
    searchLyrics,
} from '../api/client';

type Nav = NativeStackNavigationProp<RootStackParamList>;

function lyricsHitToSong(hit: LyricsSearchHit, index: number): Song {
    return {
        id: `lyrics-${index}-${hit.title}`,
        title: hit.title,
        artist: hit.artist,
        genre: 'Unknown',
        decade: 'Unknown',
        tempo: null,
    };
}

export default function FindScreen() {
    const navigation = useNavigation<Nav>();

    // lyrics search
    const [lyricsQuery, setLyricsQuery] = useState('');
    const [lyricsHits, setLyricsHits] = useState<LyricsSearchHit[] | null>(null);
    const [lyricsLoading, setLyricsLoading] = useState(false);

    // catalog filters
    const [facets, setFacets] = useState<Facets | null>(null);
    const [artist, setArtist] = useState('');
    const [genre, setGenre] = useState<string | null>(null);
    const [decade, setDecade] = useState<string | null>(null);
    const [tempo, setTempo] = useState<string | null>(null);
    const [results, setResults] = useState<Song[]>([]);
    const [resultsLoading, setResultsLoading] = useState(false);

    useEffect(() => {
        getFacets().then(setFacets).catch(() => { });
    }, []);

    const filters = useMemo(
        () => ({
            artist: artist.trim() || undefined,
            genre: genre ?? undefined,
            decade: decade ?? undefined,
            tempo: tempo ?? undefined,
        }),
        [artist, genre, decade, tempo]
    );

    useEffect(() => {
        let cancelled = false;
        setResultsLoading(true);
        const timer = setTimeout(() => {
            filterSongs(filters)
                .then((songs) => {
                    if (!cancelled) setResults(songs);
                })
                .catch(() => {
                    if (!cancelled) setResults([]);
                })
                .finally(() => {
                    if (!cancelled) setResultsLoading(false);
                });
        }, 250);
        return () => {
            cancelled = true;
            clearTimeout(timer);
        };
    }, [filters]);

    const runLyricsSearch = async () => {
        const q = lyricsQuery.trim();
        if (q.length < 2) return;
        setLyricsLoading(true);
        try {
            setLyricsHits(await searchLyrics(q));
        } catch {
            setLyricsHits([]);
        } finally {
            setLyricsLoading(false);
        }
    };

    const openSong = (song: Song) => navigation.navigate('Result', { song, method: 'tapped' });

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            <ScrollView showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
                <View style={styles.header}>
                    <Mono>find it by its features</Mono>
                    <H1 style={{ marginTop: 4 }}>Find a song</H1>
                </View>

                {/* Lyrics */}
                <H2 style={styles.sectionTitle}>Words you remember</H2>
                <Card style={styles.searchCard}>
                    <TextInput
                        value={lyricsQuery}
                        onChangeText={setLyricsQuery}
                        placeholder="…any lyrics fragment"
                        placeholderTextColor="rgba(0,0,0,0.35)"
                        style={styles.input}
                        onSubmitEditing={runLyricsSearch}
                        returnKeyType="search"
                    />
                    <Pressable onPress={runLyricsSearch} hitSlop={8}>
                        <Mono color={colors.accent}>search</Mono>
                    </Pressable>
                </Card>
                {lyricsLoading && <Mono style={styles.hint}>searching lyrics…</Mono>}
                {lyricsHits !== null && !lyricsLoading && (
                    <View style={styles.hitList}>
                        {lyricsHits.length === 0 ? (
                            <Mono style={styles.hint}>no songs found for those words</Mono>
                        ) : (
                            lyricsHits.map((hit, i) => (
                                <Pressable key={`${hit.title}-${i}`} onPress={() => openSong(lyricsHitToSong(hit, i))}>
                                    <Card style={styles.row}>
                                        <View style={{ flex: 1, gap: 2 }}>
                                            <HandText style={styles.rowTitle} numberOfLines={1}>{hit.title}</HandText>
                                            <Mono numberOfLines={1}>{hit.artist}</Mono>
                                            {hit.snippet && (
                                                <Mono numberOfLines={1} style={{ marginTop: 3, textTransform: 'none', letterSpacing: 0 }}>
                                                    “{hit.snippet}”
                                                </Mono>
                                            )}
                                        </View>
                                    </Card>
                                </Pressable>
                            ))
                        )}
                    </View>
                )}

                {/* Rhythm shortcut */}
                <H2 style={styles.sectionTitle}>Only the melody in your head?</H2>
                <Pressable onPress={() => navigation.navigate('MainTabs', { screen: 'Tap' } as any)}>
                    <Card style={styles.row}>
                        <View style={{ flex: 1 }}>
                            <HandText style={styles.rowTitle}>la-la / na-na</HandText>
                            <Mono>tap the rhythm instead →</Mono>
                        </View>
                    </Card>
                </Pressable>

                {/* Catalog filters */}
                <H2 style={styles.sectionTitle}>Browse trained songs</H2>
                <Card style={styles.searchCard}>
                    <TextInput
                        value={artist}
                        onChangeText={setArtist}
                        placeholder="Artist / vocals"
                        placeholderTextColor="rgba(0,0,0,0.35)"
                        style={styles.input}
                    />
                </Card>

                {facets && facets.genres.length > 0 && (
                    <View style={styles.chipGroup}>
                        <Mono style={styles.chipLabel}>genre</Mono>
                        <View style={styles.chipRow}>
                            {facets.genres.map((g) => (
                                <Chip key={g} label={g} on={genre === g} onPress={() => setGenre((p) => (p === g ? null : g))} />
                            ))}
                        </View>
                    </View>
                )}
                {facets && facets.decades.length > 0 && (
                    <View style={styles.chipGroup}>
                        <Mono style={styles.chipLabel}>decade</Mono>
                        <View style={styles.chipRow}>
                            {facets.decades.map((d) => (
                                <Chip key={d} label={d} on={decade === d} onPress={() => setDecade((p) => (p === d ? null : d))} />
                            ))}
                        </View>
                    </View>
                )}
                {facets && facets.tempos.length > 0 && (
                    <View style={styles.chipGroup}>
                        <Mono style={styles.chipLabel}>tempo</Mono>
                        <View style={styles.chipRow}>
                            {facets.tempos.map((t) => (
                                <Chip key={t} label={t} on={tempo === t} onPress={() => setTempo((p) => (p === t ? null : t))} />
                            ))}
                        </View>
                    </View>
                )}

                <View style={styles.hitList}>
                    {resultsLoading ? (
                        <Mono style={styles.hint}>filtering…</Mono>
                    ) : results.length === 0 ? (
                        <Mono style={styles.hint}>
                            nothing in the trained catalog matches — teach more songs!
                        </Mono>
                    ) : (
                        results.map((song) => (
                            <Pressable key={song.id} onPress={() => openSong(song)}>
                                <Card style={styles.row}>
                                    {song.albumArtUrl ? (
                                        <Image source={{ uri: song.albumArtUrl }} style={styles.art} />
                                    ) : (
                                        <HatchBox width={44} height={44} radius={9} />
                                    )}
                                    <View style={{ flex: 1, gap: 2 }}>
                                        <HandText style={styles.rowTitle} numberOfLines={1}>{song.title}</HandText>
                                        <Mono numberOfLines={1}>
                                            {song.artist} · {song.genre} · {song.decade}
                                        </Mono>
                                    </View>
                                </Card>
                            </Pressable>
                        ))
                    )}
                </View>

                <View style={{ height: 30 }} />
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 18 },
    header: { paddingTop: 10, paddingBottom: 4 },
    sectionTitle: { marginTop: 18, marginBottom: 8 },
    searchCard: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 12 },
    input: { flex: 1, fontSize: 15, color: colors.ink },
    hint: { marginTop: 10, textAlign: 'center' },
    hitList: { gap: 9, marginTop: 10 },
    row: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 10 },
    rowTitle: { fontSize: 17 },
    art: { width: 44, height: 44, borderRadius: 9 },
    chipGroup: { marginTop: 10 },
    chipLabel: { marginBottom: 6 },
    chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
});
