'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var json = require('@rollup/plugin-json');
var typescript = require('@rollup/plugin-typescript');
var commonjs = require('@rollup/plugin-commonjs');
var nodeResolve = require('@rollup/plugin-node-resolve');
var peerDepsExternal = require('rollup-plugin-peer-deps-external');

var rollup_config = {
  input: 'src/index.ts',
  output: [
    {
      dir: 'dist',
      format: 'es',
      sourcemap: true,
      preserveModules: true,
      preserveModulesRoot: 'src',
      entryFileNames: '[name].es.js',
    },
    {
      dir: 'dist',
      format: 'cjs',
      sourcemap: true,
      preserveModules: true,
      preserveModulesRoot: 'src',
      entryFileNames: '[name].cjs',
    },
  ],
  plugins: [
    // Allow importing JSON files
    json(),
    // Automatically externalize peer dependencies
    peerDepsExternal(),
    // Resolve modules from node_modules
    nodeResolve(),
    // Convert CommonJS modules to ES6
    commonjs(),
    // Compile TypeScript files and generate type declarations
    typescript({
      tsconfig: './tsconfig.build.json',
      declaration: true,
      declarationDir: 'dist/types',
      rootDir: 'src',
    }),
  ],
  // Do not bundle these external dependencies
  external: ['mongoose'],
};

exports.default = rollup_config;
