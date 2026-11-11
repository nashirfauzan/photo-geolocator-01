# photo-geolocator-01 entry — place a street photo on Earth

An entry for [make.fun](https://make.fun/bounty/photo-geolocator-01)'s "Find where a photo was taken":
the agent reads the instance's 25 street-level photos and posts a latitude and longitude for each.

## What it does

`main.ts` runs as `node main.ts` on Node 22 (erasable TypeScript, standard library only):

1. `GET $MAKEFUN_ENV_URL/photos` — the instance's 25 photos.
2. For each photo, fetch its JPEG and decide a latitude and longitude.
3. `POST $MAKEFUN_ENV_URL/answers` once, with every answer.

Two ways to place a photo, selected by the pushed secret `GEOLOC_MODE`:

| `GEOLOC_MODE` | behaviour |
| --- | --- |
| `vision` | one model call per photo through the trial's gateway (`GEOLOC_MODEL`, default `claude-haiku-5-5`), the JPEG inline as base64; a photo the model leaves without coordinates is left unanswered |
| unset | no model call at all: every photo gets `PICTURE_PRIOR`, the point that minimises the mean great-circle distance over the kit's 75 public practice photos |

The switch is a pushed secret and never a call's outcome, because the live pass and the replay must
make the same calls: a call the recording cannot answer stops the replay, which leaves the entry
unranked.

## Determinism

No wall clock, no unseeded randomness. Every request is a function of the brief, the photo's bytes
and a fixed prompt; answers are written by photo index and posted in photo order, so nothing depends
on which concurrent call came back first. A retried request is byte-identical, so the replay serves
it from the recording.

## Budget

The trial metters a call at its worst case: the request body at one token per three bytes, plus the
whole output cap, at the model's list price. A full instance of 25 photos is ~1.2 MB of JPEG, so
~544k metered input tokens: `claude-haiku-5-5` and `gpt-6-luna` ($0.10/Mtok in) come to ~$0.06 for the
instance, `gpt-6.1-terra` ($1/Mtok) ~$0.57, while `claude-sonnet-5-5` ($2/Mtok) ~$1.14 is over the
trial's $1.00. The cheap models leave the headroom; `GEOLOC_MODEL` picks one.

## Run it

```sh
# grade it on the kit's three practice instances (their answers are in the kit)
node cli/makefun.mjs grade-local --kit <kit> --bot <this repository> --driver process --instances 3
```

## Layout

```
main.ts   the entry: the env loop, the two placement modes, the answer batch
LICENSE   MIT
```

Photos come from OpenStreetView-5M (Astruc et al., CVPR 2024), street-level imagery by Mapillary
contributors, CC BY-SA 4.0; the kit's photos have their metadata segments removed and their pixels
unchanged.
