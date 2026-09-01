# AI Authoring

This project is intentionally AI-authored and AI-assisted. That is useful only
when the process makes evidence, uncertainty, verification, and review more
visible than ordinary drafting.

```text
                    BOOK ROADMAP
                         |
                         v
                    CHAPTER SPEC
                         |
                         v
                  SOURCE DISCOVERY
                         |
                         v
                  RESEARCH NOTES
                         |
                         v
                FACT / STATUS CHECK
                         |
                         v
                   CHAPTER DRAFT
                         |
             +-----------+-----------+
             |                       |
             v                       v
       CODE / EXPERIMENT        ASCII DIAGRAMS
             |                       |
             +-----------+-----------+
                         |
                         v
                  CORRECTNESS CHECK
                         |
                         v
                    TECH REVIEW
                         |
                         v
                   EDITORIAL PASS
                         |
                         v
                  CROSS-LINK CHECK
                         |
                         v
                       DONE
```

An AI agent must inspect source rather than complete a plausible story from
names. It records the repository commit and truth category, distinguishes
primary evidence from inference, and leaves open questions visible. It may not
promote planning prose, source presence, benchmark targets, or generated text
into facts without verification.

AI work should be bounded enough for another author to inspect. Research notes
are durable memory; `STATUS.md` is coordination; tests and benchmark records are
the executable evidence. Human or independent-agent technical review is still
required. No chapter is done because a model produced fluent prose.

When Codex and Claude Code collaborate, each reads `AGENTS.md`, status, roadmap,
Git state, branch/history, and the relevant chapter spec; claims one task;
updates research and status; and avoids rewriting unrelated work. Atomic
changes and explicit handoffs are preferred to sweeping regeneration.
