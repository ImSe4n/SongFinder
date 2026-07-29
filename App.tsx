import { useFonts, Gaegu_400Regular, Gaegu_700Bold } from '@expo-google-fonts/gaegu';
import * as SplashScreen from 'expo-splash-screen';
import React, { useCallback, useEffect } from 'react';
import { View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { LibraryProvider } from './src/store/LibraryContext';
import RootNavigator from './src/navigation/RootNavigator';

SplashScreen.preventAutoHideAsync().catch(() => { });

export default function App() {
    const [fontsLoaded] = useFonts({ Gaegu_400Regular, Gaegu_700Bold });

    const onLayout = useCallback(async () => {
        if (fontsLoaded) await SplashScreen.hideAsync();
    }, [fontsLoaded]);

    useEffect(() => {
        if (fontsLoaded) SplashScreen.hideAsync().catch(() => { });
    }, [fontsLoaded]);

    if (!fontsLoaded) return null;

    return (
        <SafeAreaProvider onLayout={onLayout}>
            <LibraryProvider>
                <RootNavigator />
            </LibraryProvider>
        </SafeAreaProvider>
    );
}
