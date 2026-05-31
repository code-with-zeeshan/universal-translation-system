import React, { useState } from 'react';
import { SafeAreaView, View, Text, TextInput, Button, StyleSheet } from 'react-native';
import { TranslationClient } from '@universal-translation/react-native-sdk';

const client = new TranslationClient({
  decoderUrl: 'http://localhost:8002/api/decode',
});

export default function App() {
  const [text, setText] = useState('Hello world');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  const onTranslate = async () => {
    setLoading(true);
    try {
      const res = await client.translate({ text, sourceLang: 'en', targetLang: 'es' });
      if (res.success) setOutput(res.data.translation);
      else setOutput(`Error: ${res.error?.message ?? 'Unknown error'}`);
    } catch (e: any) {
      setOutput(`Exception: ${e?.message ?? String(e)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <Text style={styles.title}>Universal Translation RN Example</Text>
      <View style={styles.row}>
        <TextInput
          style={styles.input}
          value={text}
          onChangeText={setText}
          placeholder="Enter text"
        />
        <Button title={loading ? 'Translating...' : 'Translate'} onPress={onTranslate} disabled={loading} />
      </View>
      <Text style={styles.outLabel}>Output:</Text>
      <Text style={styles.output}>{output}</Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, padding: 16 },
  title: { fontSize: 18, fontWeight: '600', marginBottom: 12 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  input: { flex: 1, borderWidth: 1, borderColor: '#ccc', borderRadius: 6, padding: 8 },
  outLabel: { marginTop: 16, fontWeight: '600' },
  output: { marginTop: 8 },
});