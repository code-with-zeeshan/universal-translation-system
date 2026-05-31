import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import mermaid from 'mermaid';

const __dirname = dirname(fileURLToPath(import.meta.url));
const inputDir = process.argv[2] || __dirname;
const outDir = process.argv[3] || join(__dirname, 'out');
const format = process.argv[4] || 'svg'; // svg only (png requires puppeteer)

mkdirSync(outDir, { recursive: true });

mermaid.initialize({
  startOnLoad: false,
  theme: 'default',
  sequence: { showSequenceNumbers: false }
});

import('jsdom').then(({ JSDOM }) => {
  const files = readFileSync(inputDir).filter(f => f.endsWith('.mmd'));

  const renderOne = async (file) => {
    const code = readFileSync(join(inputDir, file), 'utf-8');
    const base = file.replace('.mmd', '');
    try {
      const { svg } = await mermaid.render(`mermaid-${base}`, code);
      writeFileSync(join(outDir, `${base}.svg`), svg);
      console.log(`  OK  ${base}.svg`);
    } catch (err) {
      console.error(`  ERR ${base}: ${err.message}`);
    }
  };

  (async () => {
    for (const f of files) {
      console.log(`Rendering ${f}...`);
      await renderOne(f);
    }
    console.log('Done.');
  })();
});
