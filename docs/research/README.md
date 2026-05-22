# Research Workspace

Use this directory as the project's local research memory. The goal is to make
papers, GitHub references, assumptions, and implementation notes easy for humans
and coding agents to find.

## Layout

```text
docs/research/
  README.md
  source_registry.yaml      tracked metadata for papers, repos, articles
  papers/                   local PDFs and books, ignored by git
  notes/                    tracked paper/source notes
  templates/                note templates
```

## Rules

- Put PDF files in `docs/research/papers/`; this directory is intentionally ignored
  by git except for `.gitkeep`.
- Add each important paper, repo, or article to `source_registry.yaml`.
- Create one note per important source under `docs/research/notes/`.
- Keep implementation-critical conclusions in project docs or ADRs, not only in
  private paper notes.
- Avoid copying long copyrighted passages. Summarize and cite instead.

## Recommended Note Flow

1. Add source metadata to `source_registry.yaml`.
2. Copy `templates/paper_note.md` into `notes/<short_source_name>.md`.
3. Fill in:
   - why the source matters;
   - formulas or algorithms used;
   - assumptions;
   - implementation consequences;
   - open questions.
4. If the source changes architecture, add an ADR in `docs/decisions/`.
