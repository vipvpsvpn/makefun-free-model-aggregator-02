# makefun-free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/OWNER/makefun-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/OWNER/makefun-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that serves the OpenAI Chat Completions and Anthropic Messages APIs and answers every prompt with a real completion from the best live free OpenRouter model, with rating-ranked routing and failover.

## Run it

```sh
node main.ts
```

The trial harness runs `node main.ts` with `MAKEFUN_PROMPT_FILE`, `MAKEFUN_ENV_URL` and `OPENAI_BASE_URL` set; `node main.ts --list` prints the catalog the server registers, byte for byte. Node 22+, standard library only, nothing to install.

## How it works

- **Routing (quality):** every prompt is routed to the highest-rated free model in the brief (`rating_milli`) that meets the prompt's requirements — JSON mode, tools, vision, context length. JSON prompts go to the best JSON-capable model, the rest to the best model overall.
- **Failover:** a refused gateway call (429 or 5xx, a fetch error, a timeout) fails over to the next-best model that meets the prompt, and every refusal is reported on the response in the `x-makefun-upstream-failures` header. A 401/400 is not a model problem, so it fails fast instead of burning calls. A model that refuses is cooled down for the next prompt only, then tried again.
- **Catalog:** exactly the fourteen free OpenRouter ids the brief names, in the register body, in both `GET /v1/models` answers and in `--list`.
- **Both wires:** OpenAI Chat Completions and Anthropic Messages, JSON and streaming, with the dialled model id put back on every answer and every streamed chunk.
- **Keys:** no key is in this repository. The server dials the grader's gateway on the trial's dummy key; the entrant's own OpenRouter key is pushed to the competition as a secret and never enters the sandbox.

---

<sub>Entered on [make.fun](https://make.fun/?ref={referral_code}&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
