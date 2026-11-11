// photo-geolocator-01 — an entry for make.fun's "Find where a photo was taken" bounty.
//
// The agent reads the instance brief, fetches each of the 25 street-level photos from the env,
// decides a latitude and longitude for each, and posts them all in one batch. Nothing it prints is
// read: only the answers it leaves are scored.
//
// How a photo is placed. The switch is a pushed secret, never a call's outcome: a model call the live
// pass made and the replay cannot answer stops the replay, which leaves the entry unranked, so the
// mode must be settled before the first call.
//
//   GEOLOC_MODEL=claude-haiku-5-5   push this to place photos with the model gateway; the vendor key
//                                   (ANTHROPIC_API_KEY or OPENAI_API_KEY) goes in the same push
//   nothing pushed                  no model call at all: every photo gets PICTURE_PRIOR
//
// The mode rides on the model name rather than a flag of its own because a pushed value shorter than
// eight characters is refused.
//
// Two passes, and why they pay:
//   pass 1  SAMPLES prompts per photo on GEOLOC_MODEL (default claude-haiku-5-5, $0.10/Mtok in), whose
//           answers are merged with a component median: the cheap model's tail errors are what cost
//           an instance, and the median of several prompts trims them
//   pass 2  the photos pass 1 could not place, or whose samples disagree by more than
//           DISAGREE_KM, go to GEOLOC_STRONG_MODEL (default claude-sonnet-5-5, $2/Mtok in) once each,
//           most-in-doubt first and at most STRONG_MAX of them — its single call measured ~2.6x the
//           mean accuracy of the cheap model's on the kit's practice photos, and it is far too
//           expensive to run over a whole instance
//
// Budget: a call is metered by the trial at its worst case — the request body at one token per three
// bytes plus the whole output cap, at the model's list price. spent_micro_usd() mirrors that: no call
// is sent that the meter says would not fit, so the entry makes the calls it can afford and never
// spends a trial's budget on a refusal.
//
// Determinism: no clock, no unseeded randomness. Every request is a function of the photo's bytes and
// a fixed prompt; results are written by photo index, pass 2's set is chosen from those results in
// index order, and the answers are posted in photo order, so nothing depends on which concurrent
// call came back first. A retry re-sends a byte-identical request, so the replay serves it from the
// recording.
//
// Runs as `node /bot/main.ts` on Node 22 (erasable TypeScript, the standard library only).

interface Photo {
  id: string;
  url?: string;
  file?: string;
  answered?: boolean;
}

interface Point {
  lat: number;
  lon: number;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

/** The pushed model name: its presence is what turns the model path on. */
const MODEL_ENV = process.env.GEOLOC_MODEL ?? '';
const VISION = MODEL_ENV !== '';
const MODEL = MODEL_ENV === '' ? 'claude-haiku-5-5' : MODEL_ENV;
/** The model the photos pass 1 doubted go to. */
const STRONG_MODEL = process.env.GEOLOC_STRONG_MODEL ?? 'claude-sonnet-5-5';

/**
 * How many prompts each photo is asked over in pass 1, and their answers merged. Three samples of the
 * cheapest allowlisted model come to ~$0.17 an instance against the trial's $1.00; the values are
 * fixed here rather than pushed because a pushed value must be eight characters or longer.
 */
const SAMPLES = 3;
/** Photos pass 1's samples disagree about by more than this go to the strong model. */
const DISAGREE_KM = 400;
/** At most this many photos are sent to the strong model, most-in-doubt first. */
const STRONG_MAX = 10;
/** The trial caps a request's output; the cap is what the meter charges for. */
const MAX_TOKENS = 512;
const STRONG_MAX_TOKENS = 1024;
const CONCURRENCY = 5;
const STRONG_CONCURRENCY = 3;
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 2;
/** The trial's allowance, in micro-USD, and the request cap: neither is ever exceeded. */
const BUDGET_MICRO_USD = 1000000;
const REQUEST_CAP = 200;

/**
 * The list price per million tokens of the models this entry names, in micro-USD, as the kit's own
 * gateway prices them (agent-fun/pricing-rows/v1). A model the meter has no price for is not called:
 * the gateway would refuse it anyway.
 */
const PRICES: Record<string, { in: number; out: number }> = {
  'claude-haiku-5-5': { in: 100000, out: 500000 },
  'claude-sonnet-5-5': { in: 2000000, out: 10000000 },
  'claude-opus-5-5': { in: 4000000, out: 20000000 },
  'gpt-6-luna': { in: 100000, out: 500000 },
  'gpt-6-sol': { in: 2000000, out: 10000000 },
  'gpt-6-astra': { in: 10000000, out: 50000000 },
  'gpt-6.1-terra': { in: 1000000, out: 5000000 },
  'gpt-6.1-sol': { in: 2000000, out: 10000000 },
};

/**
 * The place to answer with when nothing has been read from a photo: the point that minimises the mean
 * great-circle distance over the kit's 75 public practice photos (all three partitions, whose places
 * the kit carries). Every photo is answered with something: an unanswered photo counts the maximum
 * error anyway, so even a poor guess beats leaving it blank.
 */
const PICTURE_PRIOR: [number, number] = [57.0, 27.0];

/** The prompts a photo is asked over in pass 1, in order. Each asks for the coordinates alone. */
const PROMPTS = [
  'This is a street-level photo with no metadata. Say where on Earth it was taken: your single best ' +
    'estimate of the latitude and longitude. Use the side of the road traffic drives on, road markings ' +
    'and signs, the script and language of any text, number plates, architecture, vegetation, sky and ' +
    'light. Answer with one line of JSON and nothing else: {"lat": <degrees north>, "lon": <degrees east>}',

  'You are given one street-level photograph, anywhere on Earth, with its metadata stripped. Work out ' +
    'the country first, then the region, then the nearest place you can justify, and answer with that ' +
    "place's coordinates. Reply with exactly one JSON object and no other text: " +
    '{"lat": <degrees north>, "lon": <degrees east>}',

  'From this photograph alone, name the most likely location on Earth. Weight what is unmistakable — ' +
    'driving side, road surface and markings, signage and its language and colour, plates, building ' +
    'styles, vegetation zone, terrain, and the light — and commit to the coordinates that best fit them. ' +
    'One line of JSON only: {"lat": <degrees north>, "lon": <degrees east>}',
];

/** Asked of a photo whose answers held no coordinates at all: nothing but the numbers. */
const STRICT_PROMPT =
  'Give the latitude and longitude of the place in this photo. Reply with only this JSON object and ' +
  'no words around it: {"lat": -90 to 90, "lon": -180 to 180}';

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/**
 * The coordinates an answer gives: its first JSON object holding lat and lon, else the two keys read on
 * their own (a think-style model can be cut off mid-object by its output cap), else the first two
 * numbers in range.
 */
function coordinates(text: string): Point | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: try the next object, or the keys below.
    }
  }
  const la = /"?lat(?:itude)?"?\s*[:=]\s*(-?\d+(?:\.\d+)?)/.exec(text);
  const lo = /"?lon(?:gitude|g)?"?\s*[:=]\s*(-?\d+(?:\.\d+)?)/.exec(text);
  if (la !== null && lo !== null) {
    const lat = Number(la[1]);
    const lon = Number(lo[1]);
    if (isLat(lat) && isLon(lon)) return { lat, lon };
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

/** The great-circle distance in km, on a sphere of radius 6,371 km, as the scorer measures it. */
function distanceKm(a: Point, b: Point): number {
  const rad = Math.PI / 180;
  const p1 = a.lat * rad;
  const p2 = b.lat * rad;
  const dp = p2 - p1;
  const dl = (b.lon - a.lon) * rad;
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(Math.max(0, h)))) * 6371;
}

/** The most two of the points are apart: how much the samples of one photo disagree. */
function spreadKm(points: Point[]): number {
  let most = 0;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      const d = distanceKm(points[i]!, points[j]!);
      if (d > most) most = d;
    }
  }
  return most;
}

/** The median of the values, component by component: one wild answer among several does not move it. */
function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** What the trial charges a call at its worst: the body at a token per three bytes, plus the output cap. */
function worstCaseMicroUsd(bodyBytes: number, model: string, maxTokens: number): number {
  const price = PRICES[model];
  if (price === undefined) return Number.POSITIVE_INFINITY;
  const inputTokens = Math.floor(bodyBytes / 3);
  return Math.ceil((inputTokens * price.in + maxTokens * price.out) / 1000000);
}

let spentMicroUsd = 0;
let requests = 0;

/** Whether the meter says this call fits: the only reason a call is not made. */
function affordable(bodyBytes: number, model: string, maxTokens: number): boolean {
  return requests < REQUEST_CAP && spentMicroUsd + worstCaseMicroUsd(bodyBytes, model, maxTokens) <= BUDGET_MICRO_USD;
}

/**
 * One model call, Anthropic Messages for a claude model and OpenAI chat completions for a gpt one (a
 * model on a route its vendor does not serve is refused). Returns the answer's text, or null when the
 * gateway refused the call (a refusal is final) or the meter would not pay for it.
 */
async function ask(jpegBase64: string, prompt: string, model: string, maxTokens: number): Promise<string | null> {
  const anthropic = model.startsWith('claude');
  const base = (anthropic ? need('ANTHROPIC_BASE_URL') : need('OPENAI_BASE_URL')).replace(/\/+$/, '');
  const key = anthropic ? need('ANTHROPIC_API_KEY') : need('OPENAI_API_KEY');
  const image = { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpegBase64 } };
  const text = { type: 'text', text: prompt };
  const body = JSON.stringify(
    anthropic
      ? { model, max_tokens: maxTokens, messages: [{ role: 'user', content: [image, text] }] }
      : {
          model,
          max_tokens: maxTokens,
          messages: [
            {
              role: 'user',
              content: [
                { type: 'text', text: prompt },
                { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${jpegBase64}` } },
              ],
            },
          ],
        },
  );
  const bytes = Buffer.byteLength(body);
  if (!affordable(bytes, model, maxTokens)) {
    process.stderr.write(`entry: ${model} not called: the meter holds ${BUDGET_MICRO_USD - spentMicroUsd} micro-USD\n`);
    return null;
  }
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (anthropic) {
    headers['x-api-key'] = key;
    headers['anthropic-version'] = '2023-06-01';
  } else {
    headers.authorization = `Bearer ${key}`;
  }
  for (let attempt = 0; ; attempt += 1) {
    requests += 1;
    spentMicroUsd += worstCaseMicroUsd(bytes, model, maxTokens);
    const res = await fetch(`${base}${anthropic ? '/v1/messages' : '/v1/chat/completions'}`, {
      method: 'POST',
      headers,
      body,
    });
    if (res.ok) {
      const json = (await res.json()) as {
        content?: { type: string; text?: string }[];
        choices?: { message?: { content?: string } }[];
      };
      if (anthropic) {
        return (json.content ?? [])
          .filter((b) => b.type === 'text')
          .map((b) => b.text ?? '')
          .join('\n');
      }
      return json.choices?.[0]?.message?.content ?? '';
    }
    const text = await res.text();
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`entry: the model call was refused (HTTP ${res.status}): ${text.slice(0, 200)}\n`);
    return null;
  }
  return null;
}

function photoUrl(p: Photo): string {
  const path = p.url ?? (p.file === undefined ? undefined : `/${p.file}`);
  if (path === undefined) throw new Error(`photo ${p.id} names no url`);
  return path.startsWith('http') ? path : `${envUrl}${path}`;
}

/** Runs the jobs a few at a time, writing each result by index so no order matters. */
async function pool<T>(items: T[], limit: number, run: (item: T, index: number) => Promise<void>): Promise<void> {
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < items.length) {
      const i = next;
      next += 1;
      await run(items[i]!, i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => worker()));
}

/** Pass 1: every photo, SAMPLES prompts, each answer's coordinates kept by photo index. */
async function pass1(photos: Photo[]): Promise<Point[][]> {
  const found: Point[][] = photos.map(() => []);
  if (SAMPLES > 1 && PRICES[MODEL] === undefined) {
    process.stderr.write(`entry: ${MODEL} has no price in the meter, so it is not called\n`);
    return found;
  }
  await pool(photos, CONCURRENCY, async (p, i) => {
    try {
      const jpeg = Buffer.from(await (await fetch(photoUrl(p))).arrayBuffer()).toString('base64');
      for (let s = 0; s < SAMPLES; s += 1) {
        const text = await ask(jpeg, PROMPTS[s % PROMPTS.length]!, MODEL, MAX_TOKENS);
        const c = text === null ? null : coordinates(text);
        if (c !== null) found[i]!.push(c);
        else process.stderr.write(`entry: ${p.id} sample ${s}: no coordinates in ${(text ?? '').slice(0, 120)}\n`);
      }
    } catch (err) {
      process.stderr.write(`entry: ${p.id}: ${(err as Error).message}\n`);
    }
  });
  return found;
}

/**
 * Pass 2: the photos pass 1 left unplaced, or whose samples disagree, most-in-doubt first, at most
 * STRONG_MAX of them. The choice is read off pass 1's results in photo order, so it is the same on a
 * re-run whether the calls were slow or quick.
 */
async function pass2(photos: Photo[], found: Point[][]): Promise<Point[][]> {
  const shortlist = photos
    .map((p, i) => ({ i, d: found[i]!.length === 0 ? Number.POSITIVE_INFINITY : spreadKm(found[i]!), p }))
    .filter((x) => x.d > DISAGREE_KM)
    .sort((a, b) => (b.d === a.d ? a.i - b.i : b.d - a.d))
    .slice(0, STRONG_MAX);
  if (shortlist.length === 0 || PRICES[STRONG_MODEL] === undefined || !affordable(200000, STRONG_MODEL, STRONG_MAX_TOKENS)) {
    return found;
  }
  await pool(shortlist, STRONG_CONCURRENCY, async ({ i, p, d }) => {
    try {
      const jpeg = Buffer.from(await (await fetch(photoUrl(p))).arrayBuffer()).toString('base64');
      const text = await ask(jpeg, PROMPTS[0]!, STRONG_MODEL, STRONG_MAX_TOKENS);
      const c = text === null ? null : coordinates(text);
      if (c !== null) found[i]!.push(c);
      else {
        const strict = await ask(jpeg, STRICT_PROMPT, STRONG_MODEL, STRONG_MAX_TOKENS);
        const c2 = strict === null ? null : coordinates(strict);
        if (c2 !== null) found[i]!.push(c2);
        else process.stderr.write(`entry: ${p.id} (spread ${Math.round(d)} km): the strong model placed nothing\n`);
      }
    } catch (err) {
      process.stderr.write(`entry: ${p.id}: ${(err as Error).message}\n`);
    }
  });
  return found;
}

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos?: Photo[] };
  const photos = (listed.photos ?? []).filter((p) => p.answered !== true);
  let found: Point[][] = photos.map(() => []);

  if (!VISION) {
    // No model call: the same prior for every photo, which is deterministic and needs no key.
    found = photos.map(() => [{ lat: PICTURE_PRIOR[0], lon: PICTURE_PRIOR[1] }]);
  } else {
    found = await pass1(photos);
    found = await pass2(photos, found);
  }

  const answers: Guess[] = photos.map((p, i) => {
    const points = found[i]!;
    if (points.length === 0) return { id: p.id, lat: PICTURE_PRIOR[0], lon: PICTURE_PRIOR[1] };
    if (points.length === 1) return { id: p.id, lat: points[0]!.lat, lon: points[0]!.lon };
    return { id: p.id, lat: median(points.map((c) => c.lat)), lon: median(points.map((c) => c.lon)) };
  });
  if (answers.length === 0) return 0;
  process.stderr.write(`entry: spent ${spentMicroUsd} micro-USD over ${requests} requests\n`);
  const res = await fetch(`${envUrl}/answers`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ answers }),
  });
  process.stdout.write(`entry: ${answers.length} of ${photos.length} placed, HTTP ${res.status} ${await res.text()}\n`);
  return res.ok ? 0 : 1;
}

// A top-level await would need the file to be a module, and the grader runs it as it finds it: an
// async entry point keeps `node main.ts` working with or without a package.json beside it.
void (async (): Promise<void> => {
  try {
    process.exitCode = await main();
  } catch (err) {
    process.stderr.write(`entry: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
})();
