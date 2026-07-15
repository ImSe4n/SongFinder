import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React, { useState } from 'react';
import { Alert, Pressable, StyleSheet, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, Card, H1, HandText, Mono, Wave } from '../components/UI';
import { colors } from '../theme';
import { useRhythmTap } from '../hooks/useRhythmTap';
import { RootStackParamList } from '../navigation/types';

type Nav = NativeStackNavigationProp<RootStackParamList>;

const MIN_TAPS = 4;

export default function TrainingScreen() {
    const navigation = useNavigation<Nav>();
    const [query, setQuery] = useState('');
    const { tap, reset, heights, count } = useRhythmTap();

    const canSubmit = query.trim().length > 0 && count >= MIN_TAPS;

    const submit = () => {
        Alert.alert('Thanks!', 'That rhythm will help SongFinder learn this song.');
        reset();
        setQuery('');
        navigation.goBack();
    };

    return (
        <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
            <View style={styles.header}>
                <Mono>help it learn</Mono>
                <H1 style={{ marginTop: 4 }}>Teach a rhythm</H1>
            </View>

            <Card style={styles.searchCard}>
                <View style={styles.searchIcon}>
                    <Mono>?</Mono>
                </View>
                <TextInput
                    value={query}
                    onChangeText={setQuery}
                    placeholder="Song title or artist"
                    placeholderTextColor="rgba(0,0,0,0.35)"
                    style={styles.input}
                />
            </Card>

            <View style={styles.captionBlock}>
                <HandText style={styles.caption} bold={false}>Now tap how it goes</HandText>
            </View>

            <Pressable onPress={tap} style={styles.pad}>
                <HandText style={styles.padLabel}>TAP</HandText>
                <Mono style={{ marginTop: 4 }}>the melody</Mono>
            </Pressable>

            <Wave heights={heights} height={44} />

            <View style={styles.footer}>
                <Btn label="Submit rhythm" disabled={!canSubmit} onPress={submit} />
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 18 },
    header: { paddingTop: 10, paddingBottom: 14 },
    searchCard: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 12 },
    searchIcon: { width: 26, height: 26, borderRadius: 13, borderWidth: 2, borderColor: colors.ink, alignItems: 'center', justifyContent: 'center' },
    input: { flex: 1, fontSize: 15, color: colors.ink },
    captionBlock: { alignItems: 'center', paddingVertical: 18 },
    caption: { fontSize: 18, color: 'rgba(0,0,0,0.55)' },
    pad: {
        alignSelf: 'center',
        width: 170,
        height: 170,
        borderRadius: 85,
        borderWidth: 3,
        borderStyle: 'dashed',
        borderColor: colors.ink,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 20,
    },
    padLabel: { fontSize: 22 },
    footer: { paddingVertical: 16 },
});
