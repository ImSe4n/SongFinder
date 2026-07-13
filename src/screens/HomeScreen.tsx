import { useNavigation } from '@react-navigation/native';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React from 'react';
import { Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card, H1, H2, HatchBox, Line, Mono, Wave } from '../components/UI';
import { colors } from '../theme';
import { useLibrary } from '../store/LibraryContext';
import { MainTabParamList } from '../navigation/types';
import { RootStackParamList } from '../navigation/types';
import { CompositeNavigationProp } from '@react-navigation/native';

const TAP_HEIGHTS = [40, 80, 55, 100, 35, 70];

function greeting() {
  const h = new Date().getHours();
  if (h < 12) return 'good morning';
  if (h < 18) return 'good afternoon';
  return 'good evening';
}

type Nav = CompositeNavigationProp<
  BottomTabNavigationProp<MainTabParamList, 'Home'>,
  NativeStackNavigationProp<RootStackParamList>
>;

export default function HomeScreen() {
  const navigation = useNavigation<Nav>();
  const { entries } = useLibrary();
  const recent = entries.slice(0, 3);

  return (
    <SafeAreaView style={styles.screen} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Mono>{greeting()}</Mono>
          <H1 style={styles.title}>SongFinder</H1>
        </View>

        <View style={styles.modes}>
          <Pressable onPress={() => navigation.navigate('Tap')}>
            <Card style={styles.modeCard}>
              <Mono>mode 01</Mono>
              <H2 style={styles.modeTitle}>Tap the rhythm</H2>
              <Wave heights={TAP_HEIGHTS} height={30} />
              <Line width="88%" style={{ marginTop: 4 }} />
            </Card>
          </Pressable>

          <Pressable onPress={() => navigation.navigate('Listen')}>
            <Card style={styles.modeCard}>
              <Mono>mode 02</Mono>
              <H2 style={styles.modeTitle}>Listen &amp; find</H2>
              <View style={styles.listenRow}>
                <View style={styles.playDot}>
                  <View style={styles.playInner} />
                </View>
                <Line width={undefined as any} style={{ flex: 1 }} />
              </View>
            </Card>
          </Pressable>
        </View>

        <View style={styles.recentSection}>
          <Mono style={{ marginBottom: 6 }}>recent</Mono>
          <View style={styles.recentRow}>
            {recent.length === 0 ? (
              <>
                <HatchBox width={44} height={44} radius={9} />
                <HatchBox width={44} height={44} radius={9} />
                <HatchBox width={44} height={44} radius={9} />
              </>
            ) : (
              recent.map((e) => (
                <Pressable key={e.id} onPress={() => navigation.navigate('Result', { song: e.song, method: e.method })}>
                  <HatchBox width={44} height={44} radius={9} />
                </Pressable>
              ))
            )}
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  scroll: { flexGrow: 1, paddingHorizontal: 18, paddingTop: 6 },
  header: { paddingBottom: 18 },
  title: { fontSize: 34, marginTop: 4 },
  modes: { gap: 14 },
  modeCard: { gap: 8, padding: 18 },
  modeTitle: { fontSize: 24 },
  listenRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 2 },
  playDot: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 2,
    borderColor: colors.ink,
    alignItems: 'center',
    justifyContent: 'center',
  },
  playInner: { width: 13, height: 13, borderRadius: 7, backgroundColor: colors.ink },
  recentSection: { marginTop: 26, paddingBottom: 20 },
  recentRow: { flexDirection: 'row', gap: 8 },
});
