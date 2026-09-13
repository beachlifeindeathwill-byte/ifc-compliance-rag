# Prototype Instructions

Run the local server yourself and open the preview in the browser available to this environment. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

Build app UI in `src/`. Keep `.openai/hosting.json`, `worker/index.js`, `scripts/prepare-sites-build.mjs`, and `tests/sites-worker.test.mjs` intact so the same local prototype can be handed to Sites. Before a Sites handoff, run `npm run build` and `npm run test:sites`; the build must leave `dist/client/index.html`, `dist/server/index.js`, and `dist/.openai/hosting.json`.

## Product decisions

- The visual reference is the supplied Stitch desktop workbench: dark left navigation, compact top bar, and one complete task workspace on the right.
- Regulation Q&A and BIM model review are separate primary modules. BIM model facts and fire-compliance review are separate sub-workflows after upload.
- Keep project conditions as free-form natural-language input. Suggested questions are optional accelerators, never the only way to proceed.
- Do not expose evaluation-set counts, iteration notes, retrieval internals, or UI editing controls in the end-user interface.
- Answers lead with a direct conclusion and concise basis. Source file, standard, article, PDF physical page, and original excerpt are available in collapsed evidence panels.
