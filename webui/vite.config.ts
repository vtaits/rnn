import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs';
import toml from '@iarna/toml';

function loadConfig(configPath: string) {
  if (!configPath) {
    throw new Error('CONFIG_PATH is not defined in the environment variables');
  }

  const configFile = fs.readFileSync(configPath, 'utf-8');
  const config = toml.parse(configFile);
  return config;
};

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  const config = loadConfig(env.CONFIG_PATH);

  return {
    define: {
      __APP_CONFIG__: JSON.stringify(config),
      __TRAINING_SERVER__: JSON.stringify(env.TRAINING_SERVER),
      __PREDICTION_SERVER__: JSON.stringify(env.PREDICTION_SERVER),
    },
    plugins: [react()],
  };
})
