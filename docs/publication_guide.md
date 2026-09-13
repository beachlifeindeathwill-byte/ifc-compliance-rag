# Publication Guide

This guide describes what should be included in a public repository and what should stay local.

## Include

- Application source code in `app/`, `scripts/`, and `web/`.
- Public configuration templates such as `.env.example`.
- Architecture and evaluation documentation in `docs/`.
- Compact evaluation sets in `data/eval_sets/`.
- Standard registry and applicability metadata in `data/policy/`.

## Exclude

- Raw standards and source PDFs.
- Generated chunks, extracted pages, table indexes, vector stores, embedding caches, and full retrieval runs.
- Uploaded IFC models.
- Local screenshots, visual-review notes, logs, and temporary files.
- Secrets, API keys, local absolute paths, and user-specific environment notes.

## Chunk Sharing Policy

Do not publish generated chunks by default. Chunk files usually contain large portions of source documents and are easy to treat as redistributed source text. Instead, publish:

- the chunking script,
- the chunk schema,
- a short synthetic sample if needed,
- instructions for rebuilding chunks from documents that the user is authorized to use.

## Release Checklist

1. Confirm `.env` is not tracked.
2. Confirm `data/raw_pdfs/`, `data/processed_text/`, `data/vectorstore/`, `data/structured_tables/`, `data/ifc_uploads/`, and `data/retrieval_runs/` are not tracked.
3. Run unit tests.
4. Run frontend build.
5. Review README for local paths and project-specific private wording.
6. Create a Git tag for the shared version.
