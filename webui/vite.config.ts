import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs';
import toml from '@iarna/toml';
import path from 'path';

function loadConfig(configPath: string, configDir: string | undefined) {
  if (!configPath) {
    throw new Error('CONFIG_PATH is not defined in the environment variables');
  }

  const fullPath = configDir ? path.join(configDir, configPath) : configPath;

  const configFile = fs.readFileSync(fullPath, 'utf-8');
  const config = toml.parse(configFile);
  return config;
};

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  const config = loadConfig(env.CONFIG_PATH, env.CONFIG_DIR);

  return {
    define: {
      __APP_CONFIG__: JSON.stringify(config),
      __TRAINING_SERVER__: JSON.stringify(env.TRAINING_SERVER),
      __PREDICTION_SERVER__: JSON.stringify(env.PREDICTION_SERVER),
    },
    plugins: [react()],
  };
})
