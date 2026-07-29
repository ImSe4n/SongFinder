import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import React from 'react';
import { View } from 'react-native';
import { colors } from '../theme';
import { useLibrary } from '../store/LibraryContext';
import { MainTabParamList, RootStackParamList } from './types';

import OnboardingScreen from '../screens/OnboardingScreen';
import HomeScreen from '../screens/HomeScreen';
import TapScreen from '../screens/TapScreen';
import HistoryScreen from '../screens/HistoryScreen';
import YouScreen from '../screens/YouScreen';
import ListenScreen from '../screens/ListenScreen';
import ResultScreen from '../screens/ResultScreen';
import CandidatesScreen from '../screens/CandidatesScreen';
import TrainingScreen from '../screens/TrainingScreen';
import FindScreen from '../screens/FindScreen';

const Tab = createBottomTabNavigator<MainTabParamList>();
const Stack = createNativeStackNavigator<RootStackParamList>();

function TabIcon({ focused }: { focused: boolean }) {
    return (
        <View
            style={{
                width: 15,
                height: 15,
                borderRadius: 5,
                borderWidth: 2,
                borderColor: focused ? colors.ink : 'rgba(0,0,0,0.4)',
                backgroundColor: focused ? colors.ink : 'transparent',
            }}
        />
    );
}

function MainTabs() {
    return (
        <Tab.Navigator
            screenOptions={{
                headerShown: false,
                tabBarActiveTintColor: colors.ink,
                tabBarInactiveTintColor: 'rgba(0,0,0,0.4)',
                tabBarStyle: {
                    backgroundColor: colors.card,
                    borderTopWidth: 1.5,
                    borderTopColor: colors.ink,
                    height: 78,
                    paddingTop: 8,
                },
                tabBarLabelStyle: { fontSize: 8, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.4 },
                tabBarIcon: ({ focused }) => <TabIcon focused={focused} />,
            }}
        >
            <Tab.Screen name="Home" component={HomeScreen} />
            <Tab.Screen name="Tap" component={TapScreen} />
            <Tab.Screen name="History" component={HistoryScreen} />
            <Tab.Screen name="You" component={YouScreen} />
        </Tab.Navigator>
    );
}

export default function RootNavigator() {
    const { onboardingSeen, loaded } = useLibrary();

    if (!loaded) return null;

    return (
        <NavigationContainer>
            <Stack.Navigator screenOptions={{ headerShown: false }}>
                {!onboardingSeen ? (
                    <Stack.Screen name="Onboarding" component={OnboardingScreen} />
                ) : (
                    <>
                        <Stack.Screen name="MainTabs" component={MainTabs} />
                        <Stack.Screen name="Listen" component={ListenScreen} />
                        <Stack.Screen name="Result" component={ResultScreen} />
                        <Stack.Screen name="Candidates" component={CandidatesScreen} />
                        <Stack.Screen name="Training" component={TrainingScreen} />
                        <Stack.Screen name="Find" component={FindScreen} />
                    </>
                )}
            </Stack.Navigator>
        </NavigationContainer>
    );
}
