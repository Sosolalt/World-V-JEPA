# Mini V-JEPA — Claude Instructions

You are an expert python developer, working in the state of the art studies on AI World Models.

## Project

From-scratch implementation of V-JEPA that predicts 2D billiard dynamics in **latent space** (not pixels).
Target repo: `mini-vjepa`. Full plan in [PLAN_Mini_V-JEPA.md](PLAN_Mini_V-JEPA.md).

Stack: PyTorch (MPS), pymunk, OpenCV (headless), numpy. MacBook Pro M3, float32 only.

## Domains

The project spans five distinct technical domains. Each has a dedicated testing-expert skill in [.claude/skills/](.claude/skills/):

- **physics** — billiard simulation (pymunk, energy/momentum, collisions, friction)
- **data-pipeline** — dataset generation, rendering, NPZ format
- **model-architecture** — encoder, predictor, EMA target, shapes & gradients
- **training-loop** — training loop, anti-collapse monitoring, MPS gotchas
- **evaluation** — linear probing, latent visualizations, baseline comparisons

When asked to test or validate code in one of these areas, invoke the matching skill.

## Feature review gate — MANDATORY

This project uses a multi-agent workflow. Implementation and review are separated.

**Rule:** Every time a *feature* is finished (not every few lines — a feature is a coherent unit, e.g. "physics simulator", "encoder + EMA", "training loop end-to-end", "linear probing"), spawn an **independent reviewer agent** before moving on to the next feature.

What counts as "a feature is done":
- A milestone listed in the PLAN's section 10 timeline (one of the J1–J7 deliverables), or
- A self-contained module that other modules will start depending on (e.g. `simulation/physics.py`, `mini_vjepa/encoder.py`, `scripts/train.py` end-to-end), or
- I explicitly say "this part is done".

What does **not** trigger a review:
- Renaming a variable, fixing a typo, adjusting a hyperparameter.
- Mid-feature commits, refactors-in-progress, half-written modules.

How to run the review:
1. Use the `Agent` tool to spawn a reviewer whose skill matches the domain (e.g. `physics-test-expert` reviews the physics module). The reviewer must run **independently** — give it the feature scope, the relevant file paths, and pointers to [PLAN_Mini_V-JEPA.md](PLAN_Mini_V-JEPA.md) and this CLAUDE.md. Do **not** pre-bake your own conclusions into the prompt — let the reviewer form its own.
2. The reviewer follows the **Review mode** section of its SKILL.md: understands the feature's goal in the context of the full project, checks it against the plan and CLAUDE.md, writes tests where useful, and reports findings.
3. Wait for the reviewer's verdict — **GO**, **GO with caveats**, or **NO-GO** — before starting the next feature.
4. On **NO-GO** or **GO with caveats**, address the findings, then either re-review (if the changes were substantive) or proceed (if the reviewer pre-approved the minor cleanups in writing).

Surface the verdict to me verbatim. Never silently override a NO-GO.

## Git policy — STRICT

- **Never** run `git commit` or `git push` unless I **explicitly** ask in the current message. Staging, `git status`, `git diff`, `git log` are fine when useful.
- **Never** add yourself as a contributor.
  - **No** `Co-Authored-By: Claude …` trailer.
  - **No** "Generated with Claude Code" footer.
  - **No** `🤖` or emoji attribution.
  - Author and committer must remain my identity only. Do not touch `git config user.*`.
- Commit messages must read as if I wrote them: concise, factual, no AI fingerprints.
- When I do ask for a commit, write the message in the project's existing style and stop there — do not push unless I separately say so.

## Code style

- Default to **no comments** unless the *why* is non-obvious.
- Only english
- Pin dependencies in `requirements.txt`. Don't introduce new ones without asking.
- Prefer editing existing files over creating new ones.
- No speculative abstractions; match the plan's structure in [PLAN_Mini_V-JEPA.md](PLAN_Mini_V-JEPA.md) section 7.

## Anti-collapse is load-bearing

JEPA-style training silently collapses. Any change to encoder, predictor, EMA, or loss must preserve the monitoring hooks (`avg_std`, `effective_rank`, `avg_cosine_sim`) and the variance regularization term. Flag any PR that touches these without keeping the monitors intact.
