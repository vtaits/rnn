const fs = require('fs');
const path = require('path');

const exts = ['.rs', '.cl', '.toml'];
const outputFile = 'all_sources.txt';

function getAllFiles(dir, files = []) {
  fs.readdirSync(dir).forEach(file => {
    const fullPath = path.join(dir, file);
    if (fs.statSync(fullPath).isDirectory()) {
      getAllFiles(fullPath, files);
    } else if (exts.includes(path.extname(fullPath))) {
      files.push(fullPath);
    }
  });
  return files;
}

const allFiles = getAllFiles(process.cwd());

const output = allFiles.map(f => {
  const relPath = path.relative(process.cwd(), f);
  const content = fs.readFileSync(f, 'utf8');
  return `# Файл: ${relPath}\n\n${content}\n\n`;
}).join('\n');

fs.writeFileSync(outputFile, output, 'utf8');
console.log(`Склеено ${allFiles.length} файлов в ${outputFile}`);
