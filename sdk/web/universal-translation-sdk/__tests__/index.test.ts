import { TranslationClient } from '../src/index';

describe('TranslationClient', () => {
  it('should be defined', () => {
    const client = new TranslationClient({ vocabUrl: 'https://example.com/vocab' });
    expect(client).toBeDefined();
  });
});
