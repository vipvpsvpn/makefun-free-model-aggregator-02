// free-model-aggregator-02: one server, two client wires, and a real free model behind
// every answer. Built from the house starter kit, with its three marked gaps closed:
//
//   - quality (60%): every prompt is routed to the highest-rated free model that meets its
//     requirements (rating_milli, ties broken by model id), read from the brief at runtime.
//   - failover (10%): a refused gateway call (429 or 5xx, a fetch error, a timeout) fails over
//     to the next-best model that meets the prompt, and every refusal is reported on the
//     response in the x-makefun-upstream-failures header, the only way the failover column
//     can see a refusal the server recovered from. A 401/400/402/403 is not a model problem,
//     so it fails fast instead of burning calls on other models.
//   - rate limits: a model that refuses is cooled down for the next prompt only (a
//     prompt-count cooldown, never wall-clock time), so one rate-limited model cannot take
//     every later prompt down with it, and a recovered model is tried again soon after.
//
// The catalog (5%) lists exactly the fourteen free ids the brief names, in the register
// body, in both GET /v1/models answers and in --list, byte for byte. Everything is a pure
// function of the brief, the requests and the gateway's answers: no wall clock and no
// unseeded randomness in any routing decision, so a replay leaves the same state export.
// The only clock use is the per-attempt abort timeout, which bounds liveness and never
// changes which model a prompt is served by when the gateway answers.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with Node 22's
//   standard library alone (nothing is installed; TypeScript runs directly, and a relative
//   import needs its .ts extension). The env is at $MAKEFUN_ENV_URL; your model calls go through
//   the grader's gateway at $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches
//   OpenRouter on the key you pushed to the competition. The brief at $MAKEFUN_PROMPT_FILE
//   carries two documents: the free model list (each free OpenRouter model's gateway id,
//   its published rating in milli-points, what it can do, and its free flag) and the six
//   prompts (each one's id, its api, whether it streams, its weight and the requirements a
//   model must meet to serve it). Each prompt carries a unique token the answer must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages,
//        each with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
//        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//        A process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually
//   dialled. The house records every gateway call and scores the model that really answered.
//   This server sends the brief's model id in the gateway request and puts that same id back on
//   every answer and every streamed chunk (the provider's own answer spells it without the
//   gateway's `openrouter/` prefix), so its claim is the call it made, never another.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routing, the failover, the routes.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most bytes of one gateway answer this server buffers before relaying it. */
const MAX_UPSTREAM_BYTES = 4 * 1024 * 1024;
/** One gateway attempt may take at most this long; the env caps a prompt at 30 s. */
const ATTEMPT_TIMEOUT_MS = 15000;
/** One prompt's whole failover chain may take at most this long. */
const PROMPT_BUDGET_MS = 27000;
/** The most gateway attempts spent on one prompt. */
const MAX_ATTEMPTS_PER_PROMPT = 5;
/** A default output cap, used only when a request names none (the gateway refuses uncapped calls). */
const DEFAULT_MAX_TOKENS = 1024;
/** Refusals worth failing over from: a 429 or a 5xx is the model or its provider, not the request. */
const FAILOVER_STATUSES = new Set([429, 500, 502, 503, 504]);
/** A refused model is skipped for this many following prompts, then tried again. */
const COOLDOWN_PROMPTS = 1;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/**
 * Per-model cooldowns: model id -> the first prompt index it may serve again. A model that
 * refuses on prompt i is skipped for the next COOLDOWN_PROMPTS prompts, then tried again.
 * This is a function of the prompt sequence and the gateway's answers alone, so a replay
 * takes the same path.
 */
const coolingUntil = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve a prompt, best first: the brief's rating_milli descending,
 * ties broken by model id so the order is the same on every run. A model cooling down after a
 * refusal is left out until its cooldown has passed.
 */
function rankedCandidates(requires: Requirement, promptIndex: number): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .filter((model) => (coolingUntil.get(model.id) ?? 0) <= promptIndex)
    .sort((a, b) => (b.rating_milli - a.rating_milli) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** Mark a model as cooling down: it is skipped for the next COOLDOWN_PROMPTS prompts. */
function coolDown(modelId: string, promptIndex: number): void {
  coolingUntil.set(modelId, promptIndex + 1 + COOLDOWN_PROMPTS);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): { requires: Requirement; index: number } {
  const index = manifestIndex;
  manifestIndex += 1;
  const entry = brief.prompts[index];
  if (entry !== undefined) return { requires: entry.requires, index };
  return { requires: requirementsFromBody(body), index };
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The refusal trail, as the x-makefun-upstream-failures header carries it: at most 8 rows. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

function failuresHeader(failures: readonly FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  const text = JSON.stringify(failures.slice(0, 8));
  if (Buffer.byteLength(text) > 8192) return {};
  return { 'x-makefun-upstream-failures': text };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json, timeoutMs: number): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

/** A gateway refusal's short code, read off its error body where it has one. */
function failureCode(status: number, text: string): string {
  if (text !== '') {
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed) && isRecord(parsed.error)) {
        const code = parsed.error.code;
        if (typeof code === 'string' && code !== '') return code.slice(0, 64);
        const type = parsed.error.type;
        if (typeof type === 'string' && type !== '') return type.slice(0, 64);
      }
    } catch {
      // Not JSON: fall through to the status.
    }
  }
  return `http_${status}`;
}

/** A gateway answer's whole body, buffered so a broken stream can still fail over. Null when too large. */
async function readUpstream(upstream: Response): Promise<Uint8Array | null> {
  const buffer = await upstream.arrayBuffer();
  if (buffer.byteLength > MAX_UPSTREAM_BYTES) return null;
  return new Uint8Array(buffer);
}

interface Served {
  model: string;
  /** The gateway answer's bytes, buffered whole. */
  bytes: Uint8Array;
  failures: FailureRow[];
}

interface Unserved {
  failures: FailureRow[];
}

/**
 * Dial a prompt's body through the gateway, failing over down the rating order when a model
 * refuses. `buildBody` makes the Chat Completions body for the model now being tried, so an
 * Anthropic request is re-translated for each model. Returns the model that served the prompt
 * with its answer's bytes, or the refusal trail when no model served it.
 */
async function dialWithFailover(
  buildBody: (model: string) => Json,
  requires: Requirement,
  promptIndex: number,
): Promise<Served | Unserved> {
  const candidates = rankedCandidates(requires, promptIndex);
  const failures: FailureRow[] = [];
  const deadline = Date.now() + PROMPT_BUDGET_MS;
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
    const remaining = deadline - Date.now();
    if (remaining <= 1000) break;
    attempts += 1;
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, buildBody(candidate.id), Math.min(ATTEMPT_TIMEOUT_MS, remaining));
    } catch (err) {
      const code = err instanceof Error && err.name === 'TimeoutError' ? 'timeout' : 'fetch_error';
      failures.push({ status: 0, code, model: candidate.id });
      coolDown(candidate.id, promptIndex);
      log(`prompt ${promptIndex}: ${candidate.id} ${code}, failing over`);
      continue;
    }
    if (upstream.ok) {
      const bytes = await readUpstream(upstream).catch(() => null);
      if (bytes !== null) return { model: candidate.id, bytes, failures };
      failures.push({ status: 200, code: 'answer_too_large', model: candidate.id });
      coolDown(candidate.id, promptIndex);
      continue;
    }
    const text = await upstream.text().catch(() => '');
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
    coolDown(candidate.id, promptIndex);
    log(`prompt ${promptIndex}: ${candidate.id} HTTP ${upstream.status}, failing over`);
    if (!FAILOVER_STATUSES.has(upstream.status)) break;
  }
  return { failures };
}

/** The Chat Completions body a request dials with: the output cap is never left unnamed. */
function cappedBody(body: Json): Json {
  if (typeof body.max_tokens === 'number' || typeof body.max_completion_tokens === 'number') return body;
  return { ...body, max_tokens: DEFAULT_MAX_TOKENS };
}

/** A buffered gateway answer as a stream, for the wire relays. */
function streamOf(bytes: Uint8Array): ReadableStream<Uint8Array> {
  return new Response(bytes).body as ReadableStream<Uint8Array>;
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: readonly FailureRow[]): void {
  const last = failures[failures.length - 1];
  const detail = last !== undefined ? `the gateway answered HTTP ${last.status} (${last.code})` : 'the gateway gave no answer';
  const header = failuresHeader(failures);
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', `the gateway refused every free model: ${detail}`, header);
  } else {
    sendError(res, wire, 502, 'upstream_error', `the gateway refused every free model: ${detail}`, header);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requires, index } = requirementsFor(body);
  const result = await dialWithFailover((model) => cappedBody({ ...body, model }), requires, index);
  if (!('model' in result)) return refused(res, 'openai', result.failures);
  const header = failuresHeader(result.failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...header });
    try {
      for await (const frame of openAIStreamWithModel(streamOf(result.bytes), result.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = JSON.parse(Buffer.from(result.bytes).toString('utf8'));
  } catch {
    return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with a body that is not JSON', header);
  }
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, result.model), header);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requires, index } = requirementsFor(body);
  const result = await dialWithFailover((model) => cappedBody(anthropicToOpenAI(body, model)), requires, index);
  if (!('model' in result)) return refused(res, 'anthropic', result.failures);
  const header = failuresHeader(result.failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...header });
    try {
      for await (const event of anthropicEventsFromOpenAI(streamOf(result.bytes), result.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = JSON.parse(Buffer.from(result.bytes).toString('utf8'));
  } catch {
    return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with a body that is not JSON', header);
  }
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, result.model), header);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
