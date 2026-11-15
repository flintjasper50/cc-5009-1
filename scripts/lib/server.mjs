// Minimal static server for the built package. Avoids depending on the Vite dev
// server so the release gate exercises exactly the files in dist/.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
};

export function startStaticServer(dir, { host = '127.0.0.1', port = 0 } = {}) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? '/', `http://${host}`);
    const rel = decodeURIComponent(url.pathname);
    const candidates = rel === '/' ? ['index.html'] : [rel.replace(/^\//, ''), path.join(rel.replace(/^\//, ''), 'index.html')];
    for (const candidate of candidates) {
      const file = path.join(dir, candidate);
      if (!file.startsWith(dir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) continue;
      const body = fs.readFileSync(file);
      res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
      res.end(body);
      return;
    }
    res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    res.end('not found');
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => {
      const address = server.address();
      const actualPort = typeof address === 'object' && address ? address.port : port;
      resolve({
        url: `http://${host}:${actualPort}/`,
        port: actualPort,
        close: () => new Promise((done) => server.close(() => done())),
      });
    });
  });
}
