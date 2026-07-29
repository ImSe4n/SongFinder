import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { setAudioModeAsync, useAudioPlayer, useAudioPlayerStatus } from 'expo-audio';
import React, { useEffect, useRef, useState } from 'react';
import { Alert, Image, Modal, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, Card, H1, H2, HandText, HatchBox, Mono } from '../components/UI';
import { colors } from '../theme';
import { useLibrary } from '../store/LibraryContext';
import { RootStackParamList } from '../navigation/types';
import { confirmAudio, confirmSample, getLyrics, getPreview, submitFeedback } from '../api/client';

type Nav = NativeStackNavigationProp<RootStackParamList>;
type Rt = RouteProp<RootStackParamList, 'Result'>;

export default function ResultScreen() {
    const navigation = useNavigation<Nav>();
    const { params } = useRoute<Rt>();
    const { song, method, taps, candidates, audioUri, queryLogId } = params;
    const isBrowse = params.mode === 'browse';
    const { addEntry } = useLibrary();

    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [artworkUrl, setArtworkUrl] = useState<string | null>(song.albumArtUrl ?? null);
    const player = useAudioPlayer();
    const playerStatus = useAudioPlayerStatus(player);
    const sourceLoadedRef = useRef(false);
    const playing = playerStatus.playing;

    const [lyricsVisible, setLyricsVisible] = useState(false);
    const [lyrics, setLyrics] = useState<string | null>(null);
    const [lyricsState, setLyricsState] = useState<'idle' | 'loading' | 'missing'>('idle');

    useEffect(() => {
        let cancelled = false;
        getPreview(song.title, song.artist)
            .then((res) => {
                if (cancelled) return;
                setPreviewUrl(res.previewUrl);
                if (!song.albumArtUrl && res.artworkUrl) setArtworkUrl(res.artworkUrl);
            })
            .catch(() => {});
        return () => {
            cancelled = true;
        };
    }, [song.title, song.artist]);

    const togglePlay = async () => {
        if (!previewUrl) return;
        try {
            if (!sourceLoadedRef.current) {
                await setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true });
                player.replace(previewUrl);
                sourceLoadedRef.current = true;
            }
            if (playing) {
                player.pause();
            } else {
                if (playerStatus.didJustFinish) await player.seekTo(0);
                player.play();
            }
        } catch {
            Alert.alert('Playback failed', "Couldn't play the preview clip.");
        }
    };

    const openLyrics = async () => {
        setLyricsVisible(true);
        if (lyrics || lyricsState === 'loading') return;
        setLyricsState('loading');
        try {
            const res = await getLyrics(song.title, song.artist);
            if (res.lyrics) {
                setLyrics(res.lyrics);
                setLyricsState('idle');
            } else {
                setLyricsState('missing');
            }
        } catch {
            setLyricsState('missing');
        }
    };

    const save = () => {
        addEntry(song, method);
        // Best-effort reinforcement/logging — never block or alert on these
        // failing.
        if (method === 'tapped' && taps && taps.length > 0) {
            confirmSample(song.id, taps).catch(() => {});
        }
        if (method === 'listened' && audioUri) {
            confirmAudio(song.id, audioUri).catch(() => {});
        }
        if (queryLogId) {
            submitFeedback(queryLogId, 'saved', song.id).catch(() => {});
        }
        navigation.navigate('MainTabs', { screen: 'History' } as any);
    };

    const notIt = () => {
        if (queryLogId) {
            submitFeedback(queryLogId, 'not_it').catch(() => {});
        }
        const others = (candidates ?? []).filter((c) => c.id !== song.id);
        if (others.length === 0) {
            Alert.alert('No other matches', "SongFinder doesn't have any other candidates for this rhythm yet.");
            return;
        }
        navigation.navigate('Candidates', { candidates: others, excludeId: song.id, queryLogId });
    };

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            {isBrowse && (
                <View style={styles.browseHeader}>
                    <Pressable onPress={() => navigation.goBack()} hitSlop={12}>
                        <Mono>← back</Mono>
                    </Pressable>
                </View>
            )}
            {!isBrowse && song.match != null && (
                <View style={styles.matchRow}>
                    <View style={styles.matchChip}>
                        <HandText style={styles.matchLabel}>✓ {song.match}% match</HandText>
                    </View>
                </View>
            )}

            <View style={styles.hero}>
                <View>
                    {artworkUrl ? (
                        <Image source={{ uri: artworkUrl }} style={styles.art} />
                    ) : (
                        <HatchBox width={190} height={190} radius={18}>
                            <Mono>album art</Mono>
                        </HatchBox>
                    )}
                    {previewUrl && (
                        <Pressable onPress={togglePlay} style={styles.playButton}>
                            <HandText style={styles.playIcon}>{playing ? '❚❚' : '▶'}</HandText>
                        </Pressable>
                    )}
                </View>
                <View style={{ alignItems: 'center' }}>
                    <H1 style={styles.title}>{song.title}</H1>
                    <HandText style={styles.artist} bold={false}>{song.artist}</HandText>
                </View>
            </View>

            <View style={styles.links}>
                <Pressable onPress={openLyrics}>
                    <Card style={styles.linkCard}>
                        <View style={styles.linkIconSquare} />
                        <HandText style={{ flex: 1, fontSize: 16 }} bold={false}>Read the lyrics</HandText>
                        <Mono>lyrics</Mono>
                    </Card>
                </Pressable>
                {previewUrl && (
                    <Pressable onPress={togglePlay}>
                        <Card style={styles.linkCard}>
                            <View style={styles.linkIconCircle} />
                            <HandText style={{ flex: 1, fontSize: 16 }} bold={false}>
                                {playing ? 'Pause the preview' : 'Play a 30s preview'}
                            </HandText>
                            <Mono>audio</Mono>
                        </Card>
                    </Pressable>
                )}
            </View>

            <View style={{ flex: 1 }} />

            {!isBrowse && (
                <View style={styles.footer}>
                    {(candidates?.length ?? 0) > 0 && (
                        <Btn label="Not it" ghost style={{ flex: 1 }} onPress={notIt} />
                    )}
                    <Btn label="Save to library" style={{ flex: 2 }} onPress={save} />
                </View>
            )}

            <Modal visible={lyricsVisible} animationType="slide" onRequestClose={() => setLyricsVisible(false)}>
                <SafeAreaView style={styles.lyricsScreen} edges={['top', 'bottom']}>
                    <View style={styles.lyricsHeader}>
                        <View style={{ flex: 1 }}>
                            <H2>{song.title}</H2>
                            <Mono style={{ marginTop: 2 }}>{song.artist}</Mono>
                        </View>
                        <Pressable onPress={() => setLyricsVisible(false)} hitSlop={12}>
                            <Mono>close ✕</Mono>
                        </Pressable>
                    </View>
                    <ScrollView contentContainerStyle={styles.lyricsBody}>
                        {lyricsState === 'loading' && <Mono>finding the words…</Mono>}
                        {lyricsState === 'missing' && <Mono>no lyrics found for this one</Mono>}
                        {lyrics && (
                            <HandText style={styles.lyricsText} bold={false}>{lyrics}</HandText>
                        )}
                    </ScrollView>
                </SafeAreaView>
            </Modal>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 18 },
    browseHeader: { paddingTop: 10, alignItems: 'flex-start' },
    matchRow: { alignItems: 'center', paddingTop: 10 },
    matchChip: { backgroundColor: colors.accent, borderRadius: 999, paddingVertical: 6, paddingHorizontal: 14 },
    matchLabel: { color: '#fff', fontSize: 14 },
    hero: { alignItems: 'center', gap: 14, paddingTop: 18 },
    art: { width: 190, height: 190, borderRadius: 18, borderWidth: 2, borderColor: colors.ink },
    playButton: {
        position: 'absolute',
        right: -10,
        bottom: -10,
        width: 52,
        height: 52,
        borderRadius: 26,
        backgroundColor: colors.accent,
        borderWidth: 2,
        borderColor: colors.ink,
        alignItems: 'center',
        justifyContent: 'center',
    },
    playIcon: { color: '#fff', fontSize: 18 },
    title: { fontSize: 30, textAlign: 'center' },
    artist: { fontSize: 20, color: 'rgba(0,0,0,0.55)', marginTop: 2 },
    links: { gap: 10, paddingTop: 22 },
    linkCard: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 13, paddingHorizontal: 15 },
    linkIconSquare: { width: 22, height: 22, borderRadius: 5, borderWidth: 2, borderColor: colors.ink },
    linkIconCircle: { width: 22, height: 22, borderRadius: 11, borderWidth: 2, borderColor: colors.ink },
    footer: { flexDirection: 'row', gap: 10, paddingBottom: 18 },
    lyricsScreen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 20 },
    lyricsHeader: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 14 },
    lyricsBody: { paddingBottom: 40 },
    lyricsText: { fontSize: 19, lineHeight: 28 },
});
