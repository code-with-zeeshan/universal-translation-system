const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
  entry: './src/index.ts',
  mode: 'production',
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.wasm$/,
        type: 'asset/resource',
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
  output: {
    filename: 'universal-translation-sdk.js',
    path: path.resolve(__dirname, 'dist'),
    library: 'UniversalTranslationSDK',
    libraryTarget: 'umd',
    globalObject: 'this',
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: 'public/wasm', to: 'wasm' },
      ],
    }),
  ],
  experiments: {
    asyncWebAssembly: true,
  },
  // Prevent bundling of certain imported packages and instead retrieve these external dependencies at runtime
  externals: {
    // Add any external dependencies here if needed
  },
};