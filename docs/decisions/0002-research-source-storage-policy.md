# ADR 0002: Research Source Storage Policy

Status: Accepted

## Context

The project uses academic papers, GitHub references, and future working papers.
Some source PDFs are large or copyrighted. They are useful locally for research
agents, but should not be pushed to the public repository.

## Decision

Track source metadata and implementation notes, not binary papers:

- local PDFs go under `docs/research/papers/`;
- `docs/research/papers/*` is ignored by git;
- source metadata is tracked in `docs/research/source_registry.yaml`;
- implementation notes are tracked under `docs/research/notes/`.

## Consequences

Agents can use local papers when present, while the public repository stays clean
and lightweight. A fresh clone will still have source metadata and notes, but not
the original PDFs unless the user adds them locally.
