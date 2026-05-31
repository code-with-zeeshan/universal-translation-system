// rollup.config.js
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import json from '@rollup/plugin-json';
import { terser } from 'rollup-plugin-terser';
import peerDepsExternal from 'rollup-plugin-peer-deps-external';
import dts from 'rollup-plugin-dts';
import copy from 'rollup-plugin-copy';
import { readFileSync } from 'fs';

const packageJson = JSON.parse(readFileSync('./package.json', 'utf-8'));

const external = ['onnxruntime-web'];

export default [
  // Main builds
  {
    input: 'src/index.ts',
    output: [
      {
        file: packageJson.main,
        format: 'cjs',
        sourcemap: true,
        exports: 'named'
      },
      {
        file: packageJson.module,
        format: 'esm',
        sourcemap: true
      },
      {
        file: 'dist/index.umd.js',
        format: 'umd',
        name: 'UniversalTranslation',
        sourcemap: true,
        globals: {
          'onnxruntime-web': 'ort'
        }
      }
    ],
    external,
    plugins: [
      peerDepsExternal(),
      resolve({
        browser: true,
        preferBuiltins: false,
        extensions: ['.ts', '.tsx', '.js', '.jsx']
      }),
      commonjs(),
      json(),
      typescript({ 
        tsconfig: './tsconfig.json',
        exclude: ['**/*.test.ts', '**/*.spec.ts']
      }),
      terser({
        compress: {
          drop_console: process.env.NODE_ENV === 'production'
        }
      }),
      copy({
        targets: [
          { 
            src: 'node_modules/onnxruntime-web/dist/*.wasm', 
            dest: 'dist/wasm' 
          },
          {
            src: 'public/wasm/*.wasm',
            dest: 'dist/wasm'
          },
          {
            src: 'public/wasm/*.js',
            dest: 'dist/wasm'
          },
          {
            src: 'scripts/convert-vocab.js',
            dest: 'dist/scripts'
          }
        ],
        hook: 'writeBundle'
      })
    ]
  },
  
  // React component build (separate entry)
  {
    input: 'src/components/TranslationComponent.tsx',
    output: [
      {
        file: 'dist/react.js',
        format: 'cjs',
        sourcemap: true,
        exports: 'named'
      },
      {
        file: 'dist/react.esm.js',
        format: 'esm',
        sourcemap: true
      }
    ],
    external: [...external, 'react', 'react-dom'],
    plugins: [
      peerDepsExternal(),
      resolve({
        browser: true,
        extensions: ['.ts', '.tsx', '.js', '.jsx']
      }),
      commonjs(),
      typescript({ 
        tsconfig: './tsconfig.json',
        jsx: 'react'
      }),
      terser()
    ]
  },
  
  // Type definitions
  {
    input: 'dist/types/index.d.ts',
    output: [{ file: 'dist/index.d.ts', format: 'es' }],
    plugins: [dts()],
    external: [/\.css$/]
  }
];