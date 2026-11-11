// photo-geolocator-01 — an entry for make.fun's "Find where a photo was taken" bounty.
//
// It reads the instance brief, fetches each of the 25 photos from the env, decides a latitude and
// longitude for each, and posts them in one batch. Nothing it prints is read: only the answers it
// leaves are scored.
//
// Two ways to place a photo, chosen by a pushed secret so the choice is identical in the live pass
// and in the replay (a model call that the live pass made and the replay cannot answer stops the
// replay, which leaves the entry unranked — so the mode must never depend on a call's outcome):
//
//   GEOLOC_MODE=vision   ask the model gateway about every photo, then post what it answered
//   anything else        place every photo with PICTURE_PRIOR, making no model call at all
//
// Determinism: no clock, no unseeded randomness, every request a pure function of the brief, the
// photo's bytes and the fixed prompt. Model calls run CONCURRENCY at a time but a photo's answer
// never depends on which call came back first (answers are written by index and posted in photo
// order). A retry re-sends the identical request, so the replay serves it from the recording.
//
// Runs as `node /bot/main.ts` on Node 22 (erasable TypeScript, the standard library only).

interface Photo {
  id: string;
  url?: string;
  file?: string;
  answered?: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

/** The mode switch, pushed as a secret (a non-provider name becomes the entry's own variable). */
const MODE = process.env.GEOLOC_MODE ?? '';
const VISION = MODE === 'vision';

/** The model, from the competition's allowlist. Everything on it reads images. */
const MODEL = process.env.GEOLOC_MODEL ?? 'claude-haiku-5-5';
const ANTHROPIC = true;
const MAX_TOKENS = 400;
const CONCURRENCY = 4;
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 2;

/**
 * The answer with no model call behind it: the point that minimises the mean great-circle distance
 * over the 75 photos of the kit's public practice pool (all three partitions, whose places the kit
 * carries), which is where a photo is most likely to be when nothing about it has been read.
 */
const PICTURE_PRIOR: [number, number] = [57.0, 27.0];

/**
 * The prompt. It asks for the coordinates alone and nothing else, because the instance is graded on
 * the numbers only, and a short answer keeps the output tokens (the expensive half) down.
 */
const PROMPT =
  'This is a street-level photo with no metadata. Say where on Earth it was taken: your single best ' +
  'estimate of the latitude and longitude. Think about the side of the road traffic drives on, road ' +
  'markings and signs, the script and language of any text, the plates, the architecture, the ' +
  'vegetation, the sky and the light. Answer with one line of JSON and nothing else: ' +
  '{"lat": <degrees north>, "lon": <degrees east>}';

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The coordinates an answer text gives: its first JSON object with lat and lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: try the next object, or the bare numbers below.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

/** One model call: the answer's text, or null when the gateway refused it (a refusal is final). */
async function ask(jpeg: Buffer): Promise<string | null> {
  const base = need('ANTHROPIC_BASE_URL').replace(/\/+$/, '');
  const key = need('ANTHROPIC_API_KEY');
  const body = JSON.stringify({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    const res = await fetch(`${base}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01' },
      body,
    });
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      return (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
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

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos?: Photo[] };
  const photos = (listed.photos ?? []).filter((p) => p.answered !== true);
  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);

  if (!VISION) {
    // No model call: the same prior for every photo, which is deterministic and needs no key.
    for (let i = 0; i < photos.length; i += 1) {
      guesses[i] = { id: photos[i]!.id, lat: PICTURE_PRIOR[0], lon: PICTURE_PRIOR[1] };
    }
  } else {
    let next = 0;
    const worker = async (): Promise<void> => {
      while (next < photos.length) {
        const i = next;
        next += 1;
        const p = photos[i]!;
        try {
          const jpeg = Buffer.from(await (await fetch(photoUrl(p))).arrayBuffer());
          const text = await ask(jpeg);
          if (text === null) continue;
          const c = coordinates(text);
          if (c !== null) guesses[i] = { id: p.id, lat: c.lat, lon: c.lon };
          else process.stderr.write(`entry: no coordinates for ${p.id}: ${text.slice(0, 160)}\n`);
        } catch (err) {
          process.stderr.write(`entry: ${p.id}: ${(err as Error).message}\n`);
        }
      }
    };
    await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  }

  const answers = guesses.filter((g): g is Guess => g !== null);
  if (answers.length === 0) return 0;
  const res = await fetch(`${envUrl}/answers`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ answers }),
  });
  process.stdout.write(`entry: ${answers.length} of ${photos.length} placed, HTTP ${res.status} ${await res.text()}\n`);
  return res.ok ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`entry: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
