# Measurement records

One record per measured number printed in the book (`AUTHORING.md` §5). A
record is a Markdown file named `YYYY-MM-DD-<machine>-<subject>.md`, with the
raw outputs in a sibling directory of the same name.

Each record states:

- **Date** (UTC) and the **question** the measurement answers.
- **Hardware** by model and configuration — never by host name or address.
- **Software**: OS, engine and version or commit, driver, framework.
- **Model**: name, quantization, and the file's content digest; tensor bytes.
- **Command(s)** exactly as run, with the workload: prompt and output
  lengths, batch or concurrency, repetitions and the statistic reported.
- **Conditions** that could move the result: other load, memory pressure,
  power, warm or cold caches.
- **Results**, raw outputs, and the **derived** numbers with their arithmetic.
- **Limits**: what the measurement does not show.

Raw outputs are redacted of local paths (a model file is identified by its
digest). A number that has not been reproduced by someone else is labelled a
single-machine result; conflicting later measurements are added, not
substituted.
