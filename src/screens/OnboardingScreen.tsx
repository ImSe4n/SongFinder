import React, { useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Btn, H2, HandText, Line, Mono } from '../components/UI';
import { colors } from '../theme';
import { useLibrary } from '../store/LibraryContext';

const STEPS = [
  {
    demo: (
      <View style={{ alignItems: 'center' }}>
        <HandText style={styles.demoText}>la · la · la</HandText>
        <HandText style={styles.demoBecomes}>becomes</HandText>
        <HandText style={[styles.demoText, styles.accent]}>tap · tap · tap</HandText>
      </View>
    ),
    title: 'Sing it with your finger',
    dotsOn: 0,
  },
  {
    demo: (
      <View style={styles.listenDemo}>
        <View style={styles.listenRing} />
        <View style={styles.listenDot} />
      </View>
    ),
    title: 'Or just hold up your phone',
    dotsOn: 1,
  },
  {
    demo: (
      <View style={styles.historyDemo}>
        <View style={styles.historyLine} />
        <View style={styles.historyLine} />
        <View style={[styles.historyLine, { width: '55%' }]} />
      </View>
    ),
    title: 'Every find, saved for later',
    dotsOn: 2,
  },
];

export default function OnboardingScreen({ navigation }: any) {
  const { finishOnboarding } = useLibrary();
  const [step, setStep] = useState(0);
  const isLast = step === STEPS.length - 1;
  const current = STEPS[step];

  const finish = () => {
    finishOnboarding();
  };

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
      <View style={styles.top}>
        <Mono>{step + 1} / {STEPS.length}</Mono>
        <Pressable onPress={finish}>
          <Mono>skip</Mono>
        </Pressable>
      </View>

      <View style={styles.body}>
        {current.demo}

        <View style={styles.circles}>
          {[0, 1, 2].map((i) => (
            <View key={i} style={[styles.circle, i === 2 && styles.circleAccent]} />
          ))}
        </View>

        <View style={styles.textBlock}>
          <H2 style={{ fontSize: 22, textAlign: 'center' }}>{current.title}</H2>
          <Line width="78%" style={{ marginTop: 12, alignSelf: 'center' }} />
          <Line width="60%" style={{ marginTop: 8, alignSelf: 'center' }} />
        </View>
      </View>

      <View style={styles.dots}>
        {STEPS.map((_, i) => (
          <View key={i} style={[styles.dot, i === step && styles.dotOn]} />
        ))}
      </View>

      <View style={styles.footer}>
        <Btn
          label={isLast ? 'Get started' : 'Next →'}
          size="lg"
          onPress={() => (isLast ? finish() : setStep((s) => s + 1))}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 24 },
  top: { flexDirection: 'row', justifyContent: 'space-between', paddingTop: 8 },
  body: { flex: 1, justifyContent: 'center', alignItems: 'center', gap: 28 },
  demoText: { fontSize: 34, letterSpacing: 1 },
  demoBecomes: { fontSize: 22, color: 'rgba(0,0,0,0.4)', marginVertical: 6 },
  accent: { color: colors.accent },
  circles: { flexDirection: 'row', gap: 16 },
  circle: { width: 50, height: 50, borderRadius: 25, borderWidth: 2.5, borderColor: colors.ink },
  circleAccent: { borderColor: colors.accent, backgroundColor: 'rgba(42,120,214,0.12)' },
  textBlock: { paddingHorizontal: 8 },
  listenDemo: { width: 120, height: 120, alignItems: 'center', justifyContent: 'center' },
  listenRing: { position: 'absolute', width: 120, height: 120, borderRadius: 60, borderWidth: 1.5, borderColor: 'rgba(0,0,0,0.15)' },
  listenDot: { width: 64, height: 64, borderRadius: 32, backgroundColor: colors.accent },
  historyDemo: { width: 180, gap: 10 },
  historyLine: { height: 10, borderRadius: 5, backgroundColor: colors.line, width: '85%' },
  dots: { flexDirection: 'row', justifyContent: 'center', gap: 7, paddingBottom: 18 },
  dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.dot },
  dotOn: { width: 18, backgroundColor: colors.ink },
  footer: { paddingBottom: 20 },
});
