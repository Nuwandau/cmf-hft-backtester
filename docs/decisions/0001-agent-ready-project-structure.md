# ADR 0001: Agent-Ready Project Structure

Status: Accepted

## Context

The project is expected to evolve through repeated work with coding and research
agents. Agents need fast access to project rules, domain assumptions, research
sources, and safe commands without rediscovering the whole repository.

## Decision

Add:

- root `AGENTS.md` for coding-agent instructions;
- root `CONTEXT.md` for stable quant and architecture context;
- lightweight pointers for Claude and Copilot;
- `docs/agent_memory.md` for durable project facts;
- `docs/research/` for source metadata, ignored local papers, and notes;
- `docs/decisions/` for architecture decisions.

## Consequences

Agents should start faster, preserve project invariants more reliably, and avoid
committing raw data or copyrighted papers. The tradeoff is that these files must
be kept current after major architectural or modeling changes.
