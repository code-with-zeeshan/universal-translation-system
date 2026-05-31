// scripts/convert-vocab.js
const msgpack = require('msgpack-lite');
const fs = require('fs');
const path = require('path');

function convertVocabToJSON(inputPath, outputPath) {
  try {
    console.log(`Converting ${inputPath}...`);
    
    const data = fs.readFileSync(inputPath);
    const decoded = msgpack.decode(data);
    
    // Ensure the structure matches what the web SDK expects
    const jsonData = {
      name: decoded.name,
      version: decoded.version || '1.0',
      languages: decoded.languages || [],
      tokens: decoded.tokens || {},
      subwords: decoded.subwords || {},
      special_tokens: decoded.special_tokens || decoded.specialTokens || {},
      metadata: decoded.metadata || {
        total_tokens: Object.keys(decoded.tokens || {}).length + 
                     Object.keys(decoded.subwords || {}).length + 
                     Object.keys(decoded.special_tokens || decoded.specialTokens || {}).length,
        size_mb: 0,
        coverage_percentage: decoded.metadata?.coverage_percentage,
        oov_rate: decoded.metadata?.oov_rate
      }
    };
    
    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    fs.writeFileSync(outputPath, JSON.stringify(jsonData, null, 2));
    
    const stats = fs.statSync(outputPath);
    console.log(`✅ Converted ${path.basename(inputPath)} to ${path.basename(outputPath)} (${(stats.size / 1024).toFixed(2)} KB)`);
    
  } catch (error) {
    console.error(`❌ Error converting ${inputPath}:`, error.message);
  }
}

// Main conversion
const vocabDir = path.join(__dirname, '../../vocabs');
const outputDir = path.join(__dirname, '../public/vocabs');

// Create output directory if it doesn't exist
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Convert all vocabulary packs
const files = fs.readdirSync(vocabDir);
const msgpackFiles = files.filter(file => file.endsWith('.msgpack'));

if (msgpackFiles.length === 0) {
  console.log('No .msgpack files found in', vocabDir);
  process.exit(1);
}

console.log(`Found ${msgpackFiles.length} vocabulary packs to convert\n`);

msgpackFiles.forEach(file => {
  const inputPath = path.join(vocabDir, file);
  const outputPath = path.join(outputDir, file.replace('.msgpack', '.json'));
  convertVocabToJSON(inputPath, outputPath);
});

console.log('\n✅ Vocabulary conversion complete!');