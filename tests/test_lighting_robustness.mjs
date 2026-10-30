import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {conditions, toApiValues, excelRow, latestEntries, readJournal, main} from '../scripts/lighting_robustness.mjs';

test('all 625 combinations match requested descending order and physical channel mapping', () => {
  const all = conditions();
  assert.equal(all.length, 625);
  assert.equal(new Set(all.map(c => c.join('_'))).size, 625);
  assert.deepEqual(all.slice(0, 3), [[100,100,100,100], [100,100,100,75], [100,100,100,50]]);
  assert.deepEqual(all.at(-1), [0,0,0,0]);
  assert.deepEqual(toApiValues([100,75,50,25]), [100,50,25,75]);
});

test('seven typed columns preserve true zero and leave failed measurements blank', () => {
  const entry = {condition: '100_75_50_25', status: 'ok', record: {p1: 0, p2: 50, p3: 100}, confidences: [90,80,70]};
  assert.deepEqual(excelRow(entry), ['100_75_50_25', 0,90,50,80,100,70]);
  assert.deepEqual(excelRow({...entry, status: 'measurement_failed'}), ['100_75_50_25',null,null,null,null,null,null]);
  assert.throws(() => excelRow({...entry, confidences: null}));
});

test('resume distinguishes specimens and replaces only retried conditions', () => {
  const entry = {specimen: '일반시편', condition: '0_0_0_0', index: 624, status: 'measurement_failed'};
  const result = latestEntries([entry, {...entry, specimen: '검증시편'}, {...entry, status: 'ok'}]);
  assert.equal(result.length, 2);
  assert.equal(result.find(e => e.specimen === '일반시편').status, 'ok');
});

test('a torn final checkpoint is preserved separately and earlier observations recover', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'lighting-journal-'));
  try {
    const journal = path.join(directory, 'measurements.jsonl');
    await fs.writeFile(journal, '{"index":0}\n{"index":');
    assert.deepEqual(await readJournal(journal), [{index:0}]);
    assert.equal(await fs.readFile(journal, 'utf8'), '{"index":0}\n');
    assert.equal((await fs.readdir(directory)).length, 2);
    await fs.writeFile(journal, 'bad\n{"index":0}\n');
    await assert.rejects(readJournal(journal), /손상/);
  } finally { await fs.rm(directory, {recursive:true, force:true}); }
});

test('real browser renders the project UI, checkpoints success/failure, exports seven columns, and restores lights',
  {skip: process.env.LIGHTING_BROWSER_TEST !== '1', timeout: 120000}, async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'lighting-browser-test-'));
  const project = path.resolve(import.meta.dirname, '..');
  const method = 'mean_threshold_selected_binary_label_probability_percent';
  const status = {camera: 'READY', camera_enabled: true, camera_connected: true, model: 'LOADED',
    confidence_method: method, exposure_us: 1, pixel_format: 'Mono8', threshold: 0.5,
    positive_classes: ['Exposure'], light: 'RS232 OPEN · ACK 확인', light_port: 'TEST', light_baudrate: 9600};
  const roi = {threshold: 0.25, level: 'low'};
  const applied = [];
  let inspections = 0;
  let nodeStatusReads = 0;
  const server = http.createServer(async (req, res) => {
    try {
      const route = new URL(req.url, 'http://localhost').pathname;
      if (route === '/' || route.startsWith('/static/')) {
        const file = path.join(project, 'src/web/static', route === '/' ? 'index.html' : route.slice(8));
        res.setHeader('Content-Type', file.endsWith('.html') ? 'text/html' : file.endsWith('.css') ? 'text/css' : 'text/javascript');
        res.end(await fs.readFile(file)); return;
      }
      if (route === '/api/ui/session' || route === '/api/preview') { res.statusCode = 204; res.end(); return; }
      res.setHeader('Content-Type', 'application/json');
      if (route === '/api/status') {
        if (req.headers['user-agent'] === 'node') nodeStatusReads++;
        res.end(JSON.stringify([1,2,4].includes(nodeStatusReads) ? {...status, camera:'프레임 수신 대기', camera_connected:false} : status)); return;
      }
      if (route === '/api/roi') { res.end(JSON.stringify(roi)); return; }
      if (route === '/api/bootstrap') {
        res.end(JSON.stringify({status, source:'camera', roi, history: [], lighting: {values: [100,0,0,100],
          groups: [{name:'좌판',channels:'CH1 + CH2'}, {name:'좌바',channels:'CH3'}, {name:'우바',channels:'CH4'}, {name:'우판',channels:'CH5 + CH6'}]}})); return;
      }
      const chunks = []; for await (const chunk of req) chunks.push(chunk);
      if (route === '/api/lighting/apply') {
        applied.push(JSON.parse(Buffer.concat(chunks).toString()).values);
        res.end(JSON.stringify({success:true, message:'TEST ACK', receipts:{}})); return;
      }
      if (route === '/api/inspect') {
        inspections++;
        if (inspections === 2) { res.statusCode = 400; res.end(JSON.stringify({detail:'TEST ROI detection failure'})); return; }
        const png = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6mQAAAAASUVORK5CYII=';
        res.end(JSON.stringify({mode:'normal', record:{id:'test-only',source:'camera',p1:0,p2:50,p3:100,average:50,elapsed_seconds:0.1,measured_at:'TEST'},
          fixture_padding:'x'.repeat(12 * 1024 * 1024),
          confidences:[90,80,70], confidence_method:method, valid_pixels:[100,100,100], defect_pixels:[100,50,0],
          history:[], saved_message:'TEST', images:{result:png,p1:png,p2:png,p3:png}})); return;
      }
      res.statusCode = 404; res.end('{}');
    } catch(error) {res.statusCode = 500; res.end(JSON.stringify({detail:error.message}));}
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const args = ['--project', project, '--output', directory, '--url', `http://127.0.0.1:${server.address().port}`,
      '--limit', '2', '--headless', '--no-start-server', '--settle-ms', '0'];
    await main(args);
    assert.deepEqual(applied, [[100,100,100,100], [100,100,75,100], [100,0,0,100]]);
    const entries = await readJournal(path.join(directory, 'measurements.jsonl'));
    assert.equal(entries.length, 2);
    assert.equal(entries[1].status, 'measurement_failed');
    assert.ok(nodeStatusReads >= 6, 'startup and transient camera readiness should recover before measuring');
    assert.equal((await fs.readdir(path.join(directory, '스크린샷'))).length, 2);
    const require = createRequire(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/package.json'));
    const deps = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
    const book = await deps.SpreadsheetFile.importXlsx(await deps.FileBlob.load(path.join(directory, '20261030_빛강건성_실험.xlsx')));
    const rows = book.worksheets.getItemAt(0).getRange('A1:G3').values;
    assert.deepEqual(rows[1], ['100_100_100_100',0,90,50,80,100,70]);
    assert.deepEqual(rows[2], ['100_100_100_75',null,null,null,null,null,null]);
    await main(args); // Resuming must send no experiment lighting commands.
    assert.equal(inspections, 2);
    assert.equal(applied.length, 3);
  } finally { await new Promise(resolve => server.close(resolve)); await fs.rm(directory, {recursive:true, force:true}); }
});
