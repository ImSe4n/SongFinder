import React from 'react';
import {
  Pressable,
  PressableProps,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from 'react-native';
import { colors, fonts, radii } from '../theme';

export function Line({ width = '100%', height = 8, style }: { width?: number | `${number}%`; height?: number; style?: ViewStyle }) {
  return <View style={[{ width, height, borderRadius: height / 2, backgroundColor: colors.line }, style]} />;
}

export function Mono({ children, style, color = colors.mono }: { children: React.ReactNode; style?: any; color?: string }) {
  return <Text style={[styles.mono, { color }, style]}>{children}</Text>;
}

export function HandText({ children, style, bold = true }: { children: React.ReactNode; style?: any; bold?: boolean }) {
  return <Text style={[{ fontFamily: bold ? fonts.hand : fonts.handRegular, color: colors.ink }, style]}>{children}</Text>;
}

export function H1({ children, style }: { children: React.ReactNode; style?: any }) {
  return <HandText style={[styles.h1, style]}>{children}</HandText>;
}

export function H2({ children, style }: { children: React.ReactNode; style?: any }) {
  return <HandText style={[styles.h2, style]}>{children}</HandText>;
}

export function HatchBox({ width, height, radius = radii.sm, style, children }: { width: number; height: number; radius?: number; style?: ViewStyle; children?: React.ReactNode }) {
  return (
    <View
      style={[
        {
          width,
          height,
          borderRadius: radius,
          borderWidth: 1.5,
          borderColor: colors.boxBorder,
          backgroundColor: colors.hatchA,
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
        },
        style,
      ]}
    >
      {children}
    </View>
  );
}

export function Card({ children, style }: { children: React.ReactNode; style?: ViewStyle }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

type BtnProps = PressableProps & {
  label: string;
  ghost?: boolean;
  dark?: boolean;
  size?: 'md' | 'lg';
  style?: ViewStyle;
};

export function Btn({ label, ghost, dark, size = 'md', style, ...rest }: BtnProps) {
  return (
    <Pressable
      {...rest}
      style={({ pressed }) => [
        styles.btn,
        size === 'lg' && styles.btnLg,
        ghost && (dark ? styles.btnGhostDark : styles.btnGhost),
        pressed && { opacity: 0.75 },
        style,
      ]}
    >
      <HandText style={[styles.btnLabel, ghost && (dark ? styles.btnLabelGhostDark : styles.btnLabelGhost)]}>{label}</HandText>
    </Pressable>
  );
}

export function Chip({ label, on, onPress }: { label: string; on?: boolean; onPress?: () => void }) {
  return (
    <Pressable onPress={onPress} style={[styles.chip, on && styles.chipOn]}>
      <Text style={[styles.chipLabel, on && styles.chipLabelOn]}>{label}</Text>
    </Pressable>
  );
}

export function Wave({ heights, height = 74, activeColor = colors.ink }: { heights: number[]; height?: number; activeColor?: string }) {
  return (
    <View style={[styles.wave, { height }]}>
      {heights.map((h, i) => (
        <View key={i} style={[styles.waveBar, { height: `${h}%`, backgroundColor: activeColor }]} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  mono: {
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
  },
  h1: {
    fontSize: 28,
    lineHeight: 30,
  },
  h2: {
    fontSize: 20,
  },
  card: {
    borderWidth: 2,
    borderColor: colors.ink,
    borderRadius: radii.md,
    backgroundColor: colors.card,
    padding: 12,
  },
  btn: {
    height: 48,
    borderRadius: radii.lg,
    borderWidth: 2,
    borderColor: colors.ink,
    backgroundColor: colors.ink,
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnLg: {
    height: 54,
  },
  btnGhost: {
    backgroundColor: colors.card,
  },
  btnGhostDark: {
    backgroundColor: 'transparent',
    borderColor: '#fff',
  },
  btnLabel: {
    color: '#fff',
    fontSize: 17,
  },
  btnLabelGhost: {
    color: colors.ink,
  },
  btnLabelGhostDark: {
    color: '#fff',
  },
  chip: {
    borderWidth: 1.5,
    borderColor: colors.ink,
    borderRadius: radii.pill,
    paddingVertical: 6,
    paddingHorizontal: 13,
    backgroundColor: colors.card,
  },
  chipOn: {
    backgroundColor: colors.ink,
  },
  chipLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.ink,
  },
  chipLabelOn: {
    color: '#fff',
  },
  wave: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 3,
  },
  waveBar: {
    flex: 1,
    minWidth: 3,
    borderRadius: 3,
  },
});
