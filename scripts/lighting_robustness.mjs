/**
 * Four-factor, five-level lighting experiment for the actual Vision Workbench.
 * Run with the bundled Node.js runtime; --help prints all options.
 * Excel is plain data. JSONL is the durable checkpoint, including failed trials.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

export const HEADERS = ['광량변화조건', 'P1 함침률(%)', 'P1 신뢰도(%)',
  'P2 함침률(%)', 'P2 신뢰도(%)', 'P3 함침률(%)', 'P3 신뢰도(%)'];
export const LEVELS = [100, 75, 50, 25, 0];
const METHOD = 'mean_threshold_selected_binary_label_probability_percent';
const DEFAULT_PROJECT = 'C:\\Users\\user\\Desktop\\sw\\molding_sw';
const DEFAULT_MODULES = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
const BASENAME = '20261030_빛강건성_실험';

export function conditions() {
  const result = [];
  for (const a of LEVELS) for (const b of LEVELS)
    for (const c of LEVELS) for (const d of LEVELS) result.push([a, b, c, d]);
  return result;
}

// Experiment/file order: left panel, right panel, left bar, right bar.
// Workbench/API order: left panel (CH1+2), left bar (CH3), right bar (CH4), right panel (CH5+6).
export function toApiValues([leftPanel, rightPanel, leftBar, rightBar]) {
  return [leftPanel, leftBar, rightBar, rightPanel];
}

export function excelRow(entry) {
  if (entry.status !== 'ok') return [entry.condition, null, null, null, null, null, null];
  const r = entry.record;
  const values = [r.p1, entry.confidences?.[0], r.p2, entry.confidences?.[1], r.p3, entry.confidences?.[2]];
  if (values.some(v => typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 100))
    throw new Error(`측정값 또는 신뢰도 누락/범위 오류: ${entry.condition}`);
  return [entry.condition, ...values];
}

export function latestEntries(entries) {
  const latest = new Map();
  for (const entry of entries) latest.set(`${entry.specimen}\u0000${entry.condition}`, entry);
  return [...latest.values()].sort((a, b) => a.specimen.localeCompare(b.specimen) || a.index - b.index);
}

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
async function exists(p) { try { await fs.access(p); return true; } catch { return false; } }
async function atomicJson(p, value) {
  await fs.writeFile(`${p}.tmp`, JSON.stringify(value, null, 2), 'utf8');
  await fs.rename(`${p}.tmp`, p);
}

export async function readJournal(p) {
  if (!(await exists(p))) return [];
  const content = await fs.readFile(p, 'utf8');
  const lines = content.split('\n');
  const result = [];
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    try { result.push(JSON.parse(lines[i])); }
    catch (error) {
      // A power loss can leave just the final append incomplete. Preserve it separately.
      if (i !== lines.length - 1) throw new Error(`기록 ${i + 1}행 손상: ${error.message}`);
      await fs.writeFile(`${p}.incomplete-${Date.now()}`, lines[i], 'utf8');
      await fs.writeFile(p, content.slice(0, content.lastIndexOf('\n') + 1), 'utf8');
    }
  }
  return result;
}

export async function exportExcel(entries, destination, dependencies, previewPath = null) {
  const { Workbook, SpreadsheetFile } = dependencies;
  const workbook = Workbook.create();
  const groups = new Map();
  for (const entry of latestEntries(entries)) {
    if (!groups.has(entry.specimen)) groups.set(entry.specimen, []);
    groups.get(entry.specimen).push(excelRow(entry));
  }
  if (!groups.size) groups.set('일반시편', []);
  for (const [name, rows] of groups) {
    const sheet = workbook.worksheets.add(name);
    sheet.getRange(`A1:G${rows.length + 1}`).values = [HEADERS, ...rows];
    // Fit text only; no fills, fonts, borders, filters, tables, or number formats.
    sheet.getRange('A1:G1').format.autofitColumns();
    sheet.getRange(`A1:A${rows.length + 1}`).format.autofitColumns();
  }
  workbook.recalculate();
  if (previewPath) {
    const check = await workbook.inspect({kind: 'table', range: `${[...groups.keys()][0]}!A1:G5`,
      include: 'values', tableMaxRows: 5, tableMaxCols: 7, maxChars: 1800});
    console.log(check.ndjson);
    const preview = await workbook.render({sheetName: [...groups.keys()][0], range: 'A1:G5', scale: 1.5, format: 'png'});
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  }
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  const temporary = `${destination}.tmp.xlsx`;
  await xlsx.save(temporary);
  try { await fs.rename(temporary, destination); }
  catch (error) {
    throw new Error(`엑셀 저장 실패. Excel에서 파일을 닫은 뒤 --export-only로 복구하세요. 기록은 JSONL에 보존됩니다. ${error.message}`);
  }
}

function options(argv) {
  const config = { project: DEFAULT_PROJECT, output: path.join(os.homedir(), 'Desktop', BASENAME),
    specimen: '일반시편', mode: 'normal', url: 'http://127.0.0.1:7860', settleMs: 1000,
    limit: 625, modules: process.env.MOLDING_NODE_MODULES || DEFAULT_MODULES,
    headless: false, retryFailed: false, exportOnly: false, dryRun: false, startServer: true };
  const values = {'--project': 'project', '--output': 'output', '--specimen': 'specimen',
    '--mode': 'mode', '--url': 'url', '--settle-ms': 'settleMs', '--limit': 'limit', '--modules': 'modules'};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (values[arg]) { if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error(`${arg} 값이 필요합니다.`); config[values[arg]] = argv[++i]; }
    else if (arg === '--headless') config.headless = true;
    else if (arg === '--retry-failed') config.retryFailed = true;
    else if (arg === '--export-only') config.exportOnly = true;
    else if (arg === '--dry-run') config.dryRun = true;
    else if (arg === '--no-start-server') config.startServer = false;
    else if (arg === '--help') { console.log(`일반 시편: node lighting_robustness.mjs
검증 시편: node lighting_robustness.mjs --specimen 검증시편 --mode certification
같은 이름/모드로 다시 실행하면 완료 조건을 건너뜁니다.
--project DIR --output DIR --specimen NAME --mode normal|certification
--settle-ms 1000   조명 변경 후 대기 (최소 실제 노출 2회+0.5초)
--limit 625        앞에서부터 N개 조건 (기본 전체 625)
--retry-failed     실패한 측정만 재시도 (완료된 정상 측정은 보존)
--export-only     장비 조작 없이 JSONL에서 엑셀 재생성
--dry-run         조합/채널 매핑 확인, 파일/장비 변경 없음
--headless        브라우저 창을 숨기고 동일한 UI 캡처
--url URL --modules DIR --no-start-server
실험 도중 중단: Ctrl+C 또는 결과 폴더에 STOP 파일 생성
조건 순서: 좌측판_우측판_좌측바_우측바 (Windows 파일명은 * 사용 불가)`); return null; }
    else throw new Error(`알 수 없는 옵션: ${arg}`);
  }
  if (!['normal', 'certification'].includes(config.mode)) throw new Error('mode는 normal 또는 certification이어야 합니다.');
  if (!config.specimen || config.specimen.length > 31 || /[\\/:*?"<>|\[\]\x00-\x1f]/.test(config.specimen)
      || config.specimen.endsWith('.') || config.specimen.endsWith(' ') || config.specimen.startsWith("'") || config.specimen.endsWith("'"))
    throw new Error('시편 이름은 Windows 파일명과 Excel 시트명으로 사용할 수 있는 1~31자여야 합니다.');
  config.limit = Number(config.limit); config.settleMs = Number(config.settleMs);
  if (!Number.isInteger(config.limit) || config.limit < 1 || config.limit > 625) throw new Error('limit는 1~625 정수입니다.');
  if (!Number.isFinite(config.settleMs) || config.settleMs < 0) throw new Error('settle-ms는 0 이상입니다.');
  const url = new URL(config.url);
  if (!['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) || url.protocol !== 'http:')
    throw new Error('이 실험은 로컬 비전 서버(http://127.0.0.1)에만 연결합니다.');
  return config;
}

async function getStatus(config) {
  const response = await fetch(`${config.url}/api/status`, {signal: AbortSignal.timeout(5000)});
  if (!response.ok) throw new Error(`상태 조회 HTTP ${response.status}`);
  return response.json();
}

async function ensureServer(config) {
  try { return await getStatus(config); } catch (error) { if (!config.startServer) throw error; }
  // Do not open a second process if the port belongs to a different or busy server.
  try {
    await fetch(config.url, {signal: AbortSignal.timeout(5000)});
    throw new Error('서버가 있지만 상태 API에 응답하지 않습니다. 기존 서버를 확인하세요.');
  } catch (error) { if (error.message.includes('서버가 있지만')) throw error; }
  const python = path.join(config.project, '.venv/Scripts/python.exe');
  if (!(await exists(python))) throw new Error(`프로젝트 Python이 없습니다: ${python}`);
  const log = await fs.open(path.join(config.output, 'server.log'), 'a');
  const server = spawn(python, ['main.py', '--no-browser', '--keep-server'], {
    cwd: config.project, windowsHide: true, detached: true,
    env: {...process.env, PYTHONUTF8: '1'}, stdio: ['ignore', log.fd, log.fd],
  });
  server.on('error', error => console.error('서버 시작 오류:', error.message));
  server.unref(); await log.close();
  for (let i = 0; i < 120; i++) {
    try { return await getStatus(config); } catch { await wait(500); }
  }
  throw new Error('비전 서버가 준비되지 않았습니다. server.log를 확인하세요.');
}

function protocol(status, roi) {
  return { exposure_us: status.exposure_us, pixel_format: status.pixel_format,
    threshold: status.threshold, positive_classes: status.positive_classes,
    roi_threshold: roi.threshold, roi_level: roi.level, confidence_method: status.confidence_method };
}

export async function main(argv = process.argv.slice(2)) {
  const config = options(argv);
  if (!config) return;
  const all = conditions().slice(0, config.limit);
  if (config.dryRun) {
    console.log(JSON.stringify({total: all.length, order: '좌측판_우측판_좌측바_우측바',
      channels: ['CH1+CH2', 'CH5+CH6', 'CH3', 'CH4'],
      first: all.slice(0, 6).map(c => ({condition: c.join('_'), api: toApiValues(c)})), last: all.at(-1)}, null, 2));
    return;
  }
  await fs.mkdir(config.output, {recursive: true});
  const lockPath = path.join(config.output, '.experiment.lock');
  let lock;
  try { lock = await fs.open(lockPath, 'wx'); }
  catch { throw new Error('이미 실험이 실행 중이거나 이전 잠금이 남아 있습니다. .experiment.lock의 PID가 종료되었는지 확인하세요.'); }
  await lock.writeFile(JSON.stringify({pid: process.pid, specimen: config.specimen, started: new Date().toISOString()}));
  const journal = path.join(config.output, 'measurements.jsonl');
  const xlsxPath = path.join(config.output, `${BASENAME}.xlsx`);
  const screenshotDir = path.join(config.output, '스크린샷');
  let browser, entries = [], deps, stopped = false, lightTouched = false, page, initialLighting;
  const stop = () => { stopped = true; console.log('현재 조건을 저장하고 조명을 복원한 뒤 중단합니다.'); };
  process.on('SIGINT', stop); process.on('SIGTERM', stop);
  try {
    const require = createRequire(path.join(config.modules, 'package.json'));
    deps = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
    entries = await readJournal(journal);
    if (config.exportOnly) { await exportExcel(entries, xlsxPath, deps); console.log(xlsxPath); return; }
    let status = await ensureServer(config);
    for (let attempt = 0; attempt < 50 && status.model === 'LOADED' && status.camera_enabled
      && (!status.camera_connected || status.camera !== 'READY'); attempt++) {
      await wait(500);
      status = await getStatus(config);
    }
    if (status.model !== 'LOADED' || !status.camera_enabled || !status.camera_connected || status.camera !== 'READY')
      throw new Error(`실제 카메라/모델 준비 필요: ${JSON.stringify(status)}`);
    if (status.confidence_method !== METHOD) throw new Error('신뢰도 기능을 반영하려면 비전 서버를 재시작하세요.');
    const {chromium} = require('playwright');
    const executablePath = ['C:/Program Files/Google/Chrome/Application/chrome.exe',
      'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(p => require('node:fs').existsSync(p));
    browser = await chromium.launch({headless: config.headless, ...(executablePath ? {executablePath} : {})});
    const context = await browser.newContext({viewport: {width: 1920, height: 1400}, deviceScaleFactor: 1});
    context.setDefaultTimeout(120000);
    page = await context.newPage();
    await page.goto(config.url, {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => document.querySelector('#lighting-groups input') && document.querySelector('#system-status').textContent.includes('MODEL'));
    await page.locator('#source').selectOption('camera');
    await page.locator(`input[name="mode"][value="${config.mode}"]`).check();
    const bootstrapResponse = await page.request.get(`${config.url}/api/bootstrap`);
    const bootstrap = await bootstrapResponse.json();
    initialLighting = bootstrap.lighting.values;
    const initialProtocol = protocol(status, bootstrap.roi);
    const manifestPath = path.join(config.output, 'experiment.json');
    const manifest = await exists(manifestPath) ? JSON.parse(await fs.readFile(manifestPath, 'utf8')) : {version: 1, specimens: {}};
    const previous = manifest.specimens[config.specimen];
    if (previous && (previous.mode !== config.mode || JSON.stringify(previous.protocol) !== JSON.stringify(initialProtocol)))
      throw new Error('기존 시편 기록과 검사 모드/촬영/ROI 설정이 다릅니다. 설정을 복원하거나 새 --specimen 이름을 사용하세요.');
    manifest.specimens[config.specimen] = {...previous, mode: config.mode, protocol: initialProtocol,
      started_at: previous?.started_at || new Date().toISOString(),
      order: ['left_panel', 'right_panel', 'left_bar', 'right_bar'], levels: LEVELS, total: 625,
      restore_lighting_api_order: initialLighting,
      confidence_definition: '100 × mean(p if p >= threshold else 1-p), valid measured pixels only; model certainty, not calibrated accuracy',
      failed_trial: 'six numeric cells left blank; details in measurements.jsonl and screenshot',
      settings_snapshot: await fs.readFile(path.join(config.project, 'config/preprocess.yaml'), 'utf8'),
      roi_snapshot: await fs.readFile(path.join(config.project, 'config/roi_gabor_boundary.yaml'), 'utf8')};
    await atomicJson(manifestPath, manifest);
    await fs.mkdir(screenshotDir, {recursive: true});
    await page.evaluate(() => {
      // Camera responses contain several full-resolution PNGs and can exceed
      // Chromium's inspector response cache. Read in the page before returning
      // the response to the existing UI; only the small numeric payload crosses CDP.
      const originalFetch = window.fetch.bind(window);
      window.__experimentLastResponse = null;
      window.fetch = async (...args) => {
        const response = await originalFetch(...args);
        const input = args[0];
        if (new URL(typeof input === 'string' ? input : input.url, location.href).pathname === '/api/inspect') {
          const data = await response.clone().json();
          window.__experimentLastResponse = {ok: response.ok, status: response.status,
            payload: {record: data.record, mode: data.mode, confidences: data.confidences,
              confidence_method: data.confidence_method, detail: data.detail}};
        }
        return response;
      };
      const banner = document.createElement('div'); banner.id = 'experiment-condition';
      banner.style.cssText = 'padding:12px 20px;background:#18222c;color:white;font:16px sans-serif;white-space:pre-wrap';
      document.body.prepend(banner);
      document.querySelectorAll('input,select,button,a').forEach(el => {
        if (el.id !== 'inspect') { el.disabled = true; el.style.pointerEvents = 'none'; }
      });
    });
    const latest = latestEntries(entries).filter(e => e.specimen === config.specimen);
    const completed = new Set(latest.filter(e => !config.retryFailed || e.status === 'ok').map(e => e.condition));
    const settle = Math.max(config.settleMs, status.exposure_us / 1000 * 2 + 500);
    console.log(`${config.specimen} / ${config.mode} / ${all.length}조건 / 안정화 ${settle}ms / 기존 ${completed.size}건`);
    let added = 0;
    for (let index = 0; index < all.length; index++) {
      if (stopped || await exists(path.join(config.output, 'STOP'))) { stopped = true; break; }
      const condition = all[index].join('_');
      if (completed.has(condition)) continue;
      let currentStatus = await getStatus(config);
      // GigE can briefly reconnect between trials. Never measure until fresh
      // acquisition is READY; abort if it does not recover within three seconds.
      for (let attempt = 0; attempt < 6 && (!currentStatus.camera_connected || currentStatus.camera !== 'READY'); attempt++) {
        await wait(500);
        currentStatus = await getStatus(config);
      }
      const roi = await (await page.request.get(`${config.url}/api/roi`)).json();
      if (!currentStatus.camera_connected || currentStatus.camera !== 'READY' || currentStatus.model !== 'LOADED')
        throw new Error(`카메라/모델 준비 실패: ${JSON.stringify(currentStatus)}`);
      if (JSON.stringify(protocol(currentStatus, roi)) !== JSON.stringify(initialProtocol))
        throw new Error(`실험 설정 변경 감지: ${JSON.stringify({expected: initialProtocol, current: protocol(currentStatus, roi)})}`);
      const apiValues = toApiValues(all[index]);
      lightTouched = true;
      const applied = await page.request.post(`${config.url}/api/lighting/apply`, {data: {values: apiValues}});
      const receipt = await applied.json();
      if (!applied.ok() || !receipt.success) throw new Error(`조명 ACK 실패 (${condition}): ${receipt.message || JSON.stringify(receipt)}`);
      await page.evaluate(({values, message, title}) => {
        updateLightingValues(values);
        document.querySelector('#lighting-message').textContent = message;
        document.querySelector('#experiment-condition').textContent = title;
      }, {values: apiValues, message: receipt.message,
        title: `${config.specimen} · ${index + 1}/625 · 좌측판 ${all[index][0]}% / 우측판 ${all[index][1]}% / 좌측바 ${all[index][2]}% / 우측바 ${all[index][3]}%`});
      await page.waitForTimeout(settle);
      await page.evaluate(() => { window.__experimentLastResponse = null; });
      await page.locator('#inspect').click();
      await page.waitForFunction(() => window.__experimentLastResponse && !document.querySelector('#inspect').disabled);
      const response = await page.evaluate(() => window.__experimentLastResponse);
      const payload = response.payload;
      let entry = {specimen: config.specimen, mode: config.mode, index, condition,
        lighting: all[index], api_values: apiValues, measured_at: new Date().toISOString(), receipt};
      if (response.ok) {
        if (payload.mode !== config.mode || payload.record?.source !== 'camera')
          throw new Error('검사 모드/입력이 실험 조건과 일치하지 않습니다.');
        entry = {...entry, status: 'ok', record: payload.record, confidences: payload.confidences,
          confidence_method: payload.confidence_method};
        excelRow(entry); // Missing confidences are an infrastructure error, never silently imputed.
        await page.waitForFunction(() => ['result-image', 'p1-image', 'p2-image', 'p3-image'].every(id => {
          const image = document.getElementById(id); return !image.hidden && image.complete && image.naturalWidth > 0;
        }));
      } else {
        if (response.status !== 400) throw new Error(`검사 서버 HTTP ${response.status}: ${JSON.stringify(payload)}`);
        const health = await getStatus(config);
        if (!health.camera_connected || health.camera !== 'READY' || health.model !== 'LOADED') throw new Error(`장비/모델 오류: ${JSON.stringify(health)}`);
        entry = {...entry, status: 'measurement_failed', error: payload.detail};
      }
      const screenshot = `${config.specimen}_${condition}${entry.status === 'ok' ? '' : '_측정실패'}.png`;
      entry.screenshot = path.join('스크린샷', screenshot);
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({path: path.join(screenshotDir, `${screenshot}.tmp`), type: 'png', fullPage: true, animations: 'disabled'});
      await fs.rename(path.join(screenshotDir, `${screenshot}.tmp`), path.join(screenshotDir, screenshot));
      const handle = await fs.open(journal, 'a');
      try { await handle.writeFile(`${JSON.stringify(entry)}\n`); await handle.sync(); } finally { await handle.close(); }
      entries.push(entry); added++;
      if (added === 1 || added % 10 === 0) await exportExcel(entries, xlsxPath, deps);
      const values = entry.status === 'ok' ? [entry.record.p1, entry.record.p2, entry.record.p3].map(n => n.toFixed(4)).join(', ') : entry.error;
      console.log(`[${index + 1}/625] ${condition} ${entry.status} ${values}`);
    }
    const selected = latestEntries(entries).filter(e => e.specimen === config.specimen);
    const summary = {specimen: config.specimen, mode: config.mode, completed: selected.length,
      success: selected.filter(e => e.status === 'ok').length,
      failed: selected.filter(e => e.status !== 'ok').length, total: 625, stopped,
      finished_at: new Date().toISOString(), xlsx: xlsxPath};
    await atomicJson(path.join(config.output, `summary_${config.specimen}.json`), summary);
    console.log(JSON.stringify(summary, null, 2));
  } finally {
    // Restore the saved four-group setting after completion, interruption or failure.
    // The controller cannot read back pre-experiment hardware brightness.
    try {
      if (lightTouched && initialLighting) {
        const restored = await fetch(`${config.url}/api/lighting/apply`, {method: 'POST',
          headers: {'Content-Type': 'application/json'}, body: JSON.stringify({values: initialLighting}),
          signal: AbortSignal.timeout(15000)});
        const result = await restored.json();
        if (!restored.ok || !result.success) throw new Error(JSON.stringify(result));
        console.log(`저장 조명 설정 복원 ACK 확인: ${initialLighting.join(',')}`);
      }
    } catch (error) { console.error(`조명 복원 실패: ${error.message}`); process.exitCode = 1; }
    try { if (deps && entries.length) await exportExcel(entries, xlsxPath, deps); }
    finally {
      try { await browser?.close(); }
      finally {
        await lock.close(); await fs.unlink(lockPath);
        process.off('SIGINT', stop); process.off('SIGTERM', stop);
      }
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1; });
}
