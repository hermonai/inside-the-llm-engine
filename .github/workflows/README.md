# Workflow Policy

Phase 0 deliberately does not install a large documentation toolchain. When the
first Rust/Python/C artifacts land, add lightweight CI for formatting, tests,
structure/link checks, and any canonical diagram check. CI must not download
large model weights or silently skip required release-gate fixtures.
