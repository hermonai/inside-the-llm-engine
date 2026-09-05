# Chapter visual storyboards

Each row is an ordered sequence, not a requirement to create one overloaded
figure. Existing chapter text and fixtures remain authoritative until the pass
has met all parity gates.

| Chapter / zoom | Ordered visual explanation | Experiment and synthesis |
| --- | --- | --- |
| 1 / product → engine | request/model/token; add scheduler; add stream; lifecycle sequence; terminal states; control/data separation | cancel before token commit; one terminal owner |
| 2 / engine → bytes | Unicode scalars; bytes; ranked BPE pairs; special IDs; template; split UTF-8 emission | partial scalar; model/tokenizer contract |
| 3 / model → operator | token ID; selected D=3 row; one row dot; all logits; shared weights/local activations | one-weight intervention; same-last-ID limitation |
| 4 / operator → runtime | logits; temperature; filters; CDF; draw; token commit; feedback; cancellation | fixed-draw oracle; before/after commit |
| 5 / tensor → memory | A [2,3]; strides; index to byte; transpose; copy; reshape; ownership UML; rejection | offset oracle; owner/view/copy synthesis |
| 6 / operator → kernel | row selection; matched products; accumulator; whole matrix; strides; loop order; tiling; measured comparison | tail tile differential; retain historical negative benchmark |
| 7 / model → operator | ID row; owned x; square; sum; mean; epsilon; inverse root; scale; two passes; F32 failures | epsilon and overflow stress; normalized residual boundary |
| 8 / operator → memory | normalized residual; independent Wq/Wk/Wv; one projection row; three outputs; head grouping; GQA widths | oracle and typed errors; position-blind boundary |
| 9 / operator → geometry | pair; coordinate plane; angle; rotation matrix; positions; Q/K relative dot product | norm and relative-position oracle; no value rotation |
| 10 / block → operator | residual; QKV; heads; position; dot; scale; mask; softmax; weighted V; concat; output projection; residual | dense attention oracle; cache is introduced only after recomputation semantics |

The ten first-milestone prototypes live in the [atlas](ATLAS.md). The later
chapter prototypes are design probes, not chapter completion evidence.
