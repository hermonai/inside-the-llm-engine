# Authoring Workflow

The workflow is evidence-first and conflict-resistant.

```text
ROADMAP ──▶ CHAPTER SPEC ──▶ SOURCE DISCOVERY ──▶ RESEARCH NOTE
        ──▶ FACT/STATUS CHECK ──▶ DRAFT ──▶ CODE + UNICODE TEXT DIAGRAMS
        ──▶ CORRECTNESS ──▶ TECH REVIEW ──▶ EDITORIAL REVIEW
        ──▶ CROSS-LINK/TERMINOLOGY CHECK ──▶ COMPLETE
```

## Starting a task

Read `AGENTS.md`, `STATUS.md`, `ROADMAP.md`, the chapter specification, Git
status, current branch, and recent history. Choose one bounded task. Update its
status to RESEARCHING only when active work and a research note exist.

## Research and outline

Answer the chapter's key question, inspect primary sources, record code paths
and commits, classify system claims, list open questions, and propose diagrams,
experiments, and correctness gates. Refine the outline when evidence contradicts
it; record structural changes rather than silently reshaping the curriculum.

## Draft and implementation

Write the problem and mental model first. Build the smallest correct reference,
then expose its failure at scale, then implement the production-shaped design.
Keep code, diagrams, research, and prose in the same change when practical, but
use atomic commits that a second agent can review or continue.

## Review gates

- **Correctness:** tests and independent oracle pass.
- **Technical:** claims, shapes, ownership, code, and status are accurate.
- **Editorial:** progression, tone, terminology, and exercises teach clearly.
- **Cross-link:** references, glossary, diagrams, labs, milestones, and next
  chapter assumptions agree.

Status may advance through PLANNED, RESEARCHING, OUTLINED, DRAFTING,
CODE-COMPLETE, TECH-REVIEW, EDIT-REVIEW, COMPLETE. Skipping a label requires a
recorded reason; COMPLETE never means “prose drafted.”

## Collaboration

Agents must avoid unrelated rewrites, preserve others' work, and leave a useful
research/status handoff. If current evidence is ambiguous, record the ambiguity
and narrow the claim. `STATUS.md` names the next recommended task so a fresh
session can continue without chat history.
