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
