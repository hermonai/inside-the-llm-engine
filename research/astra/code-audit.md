# Code example audit

Source examples and oracles are executable; manuscript fences are
classified separately as excerpts, command recipes or non-executable
representations. Rust build/test/Clippy checks do not prove isolated
fences compile. Similar oracle arithmetic is intentional independence.

| Source | Role | Decision / validation |
| --- | --- | --- |
| [chapter04_sampling_cost.rs](../../code/mini-engine/crates/engine0/examples/chapter04_sampling_cost.rs) | experiment | KEEP; run directly; no competing implementation layer |
| [chapter05_traversal.rs](../../code/mini-engine/crates/engine0/examples/chapter05_traversal.rs) | experiment | KEEP; run directly; no competing implementation layer |
| [chapter06_bench.rs](../../code/mini-engine/crates/engine0/examples/chapter06_bench.rs) | experiment | KEEP; run directly; no competing implementation layer |
| [chapter07_scale_and_stress.rs](../../code/mini-engine/crates/engine0/examples/chapter07_scale_and_stress.rs) | experiment | KEEP; run directly; no competing implementation layer |
| [chapter03_oracle.py](../../code/reference/python/chapter03_oracle.py) | independent oracle | KEEP; run directly; no competing implementation layer |
| [chapter04_sampling_oracle.py](../../code/reference/python/chapter04_sampling_oracle.py) | independent oracle | KEEP; run directly; no competing implementation layer |
| [chapter05_tensor_oracle.py](../../code/reference/python/chapter05_tensor_oracle.py) | independent oracle | KEEP; run directly; no competing implementation layer |
| [chapter06_matmul_oracle.py](../../code/reference/python/chapter06_matmul_oracle.py) | independent oracle | KEEP; run directly; no competing implementation layer |
| [chapter07_embedding_rmsnorm_oracle.py](../../code/reference/python/chapter07_embedding_rmsnorm_oracle.py) | independent oracle | KEEP; run directly; no competing implementation layer |

## Manuscript fence inventory

| Chapter / line | Language | Classification |
| --- | --- | --- |
| 1:38 | text | diagram/data/pseudocode; not an executable program |
| 1:64 | text | diagram/data/pseudocode; not an executable program |
| 1:174 | text | diagram/data/pseudocode; not an executable program |
| 1:256 | text | diagram/data/pseudocode; not an executable program |
| 1:338 | text | diagram/data/pseudocode; not an executable program |
| 1:680 | text | diagram/data/pseudocode; not an executable program |
| 1:686 | text | diagram/data/pseudocode; not an executable program |
| 1:708 | rust | reference excerpt; covered by workspace, not standalone |
| 1:729 | rust | reference excerpt; covered by workspace, not standalone |
| 1:772 | text | diagram/data/pseudocode; not an executable program |
| 1:795 | sh | command recipe; environment/workload dependent |
| 1:826 | sh | command recipe; environment/workload dependent |
| 2:7 | text | diagram/data/pseudocode; not an executable program |
| 2:46 | text | diagram/data/pseudocode; not an executable program |
| 2:55 | text | diagram/data/pseudocode; not an executable program |
| 2:97 | text | diagram/data/pseudocode; not an executable program |
| 2:151 | text | diagram/data/pseudocode; not an executable program |
| 2:180 | text | diagram/data/pseudocode; not an executable program |
| 2:264 | text | diagram/data/pseudocode; not an executable program |
| 2:274 | text | diagram/data/pseudocode; not an executable program |
| 2:304 | text | diagram/data/pseudocode; not an executable program |
| 2:454 | rust | reference excerpt; covered by workspace, not standalone |
| 2:479 | text | diagram/data/pseudocode; not an executable program |
| 2:492 | text | diagram/data/pseudocode; not an executable program |
| 2:499 | text | diagram/data/pseudocode; not an executable program |
| 2:515 | text | diagram/data/pseudocode; not an executable program |
| 2:564 | text | diagram/data/pseudocode; not an executable program |
| 2:599 | text | diagram/data/pseudocode; not an executable program |
| 2:617 | text | diagram/data/pseudocode; not an executable program |
| 2:645 | text | diagram/data/pseudocode; not an executable program |
| 2:664 | text | diagram/data/pseudocode; not an executable program |
| 2:670 | text | diagram/data/pseudocode; not an executable program |
| 2:684 | rust | reference excerpt; covered by workspace, not standalone |
| 2:723 | text | diagram/data/pseudocode; not an executable program |
| 2:740 | sh | command recipe; environment/workload dependent |
| 2:810 | sh | command recipe; environment/workload dependent |
| 2:865 | text | diagram/data/pseudocode; not an executable program |
| 3:13 | text | diagram/data/pseudocode; not an executable program |
| 3:36 | text | diagram/data/pseudocode; not an executable program |
| 3:136 | text | diagram/data/pseudocode; not an executable program |
| 3:170 | text | diagram/data/pseudocode; not an executable program |
| 3:177 | text | diagram/data/pseudocode; not an executable program |
| 3:198 | text | diagram/data/pseudocode; not an executable program |
| 3:243 | text | diagram/data/pseudocode; not an executable program |
| 3:257 | rust | reference excerpt; covered by workspace, not standalone |
| 3:287 | text | diagram/data/pseudocode; not an executable program |
| 3:307 | text | diagram/data/pseudocode; not an executable program |
| 3:328 | text | diagram/data/pseudocode; not an executable program |
| 3:338 | text | diagram/data/pseudocode; not an executable program |
| 3:406 | sh | command recipe; environment/workload dependent |
| 3:412 | text | diagram/data/pseudocode; not an executable program |
| 3:485 | text | diagram/data/pseudocode; not an executable program |
| 3:494 | text | diagram/data/pseudocode; not an executable program |
| 3:500 | text | diagram/data/pseudocode; not an executable program |
| 3:524 | text | diagram/data/pseudocode; not an executable program |
| 3:552 | text | diagram/data/pseudocode; not an executable program |
| 3:579 | text | diagram/data/pseudocode; not an executable program |
| 3:725 | text | diagram/data/pseudocode; not an executable program |
| 3:734 | text | diagram/data/pseudocode; not an executable program |
| 3:769 | rust | reference excerpt; covered by workspace, not standalone |
| 3:785 | rust | reference excerpt; covered by workspace, not standalone |
| 3:810 | text | diagram/data/pseudocode; not an executable program |
| 3:856 | text | diagram/data/pseudocode; not an executable program |
| 3:868 | text | diagram/data/pseudocode; not an executable program |
| 3:896 | text | diagram/data/pseudocode; not an executable program |
| 3:925 | sh | command recipe; environment/workload dependent |
| 3:964 | text | diagram/data/pseudocode; not an executable program |
| 3:1027 | text | diagram/data/pseudocode; not an executable program |
| 3:1035 | text | diagram/data/pseudocode; not an executable program |
| 3:1041 | text | diagram/data/pseudocode; not an executable program |
| 3:1062 | text | diagram/data/pseudocode; not an executable program |
| 3:1158 | text | diagram/data/pseudocode; not an executable program |
| 3:1167 | text | diagram/data/pseudocode; not an executable program |
| 3:1185 | text | diagram/data/pseudocode; not an executable program |
| 3:1197 | text | diagram/data/pseudocode; not an executable program |
| 4:5 | text | diagram/data/pseudocode; not an executable program |
| 4:19 | text | diagram/data/pseudocode; not an executable program |
| 4:49 | rust | reference excerpt; covered by workspace, not standalone |
| 4:120 | text | diagram/data/pseudocode; not an executable program |
| 4:209 | text | diagram/data/pseudocode; not an executable program |
| 4:237 | text | diagram/data/pseudocode; not an executable program |
| 4:278 | text | diagram/data/pseudocode; not an executable program |
| 4:286 | text | diagram/data/pseudocode; not an executable program |
| 4:296 | rust | reference excerpt; covered by workspace, not standalone |
| 4:350 | text | diagram/data/pseudocode; not an executable program |
| 4:391 | text | diagram/data/pseudocode; not an executable program |
| 4:415 | text | diagram/data/pseudocode; not an executable program |
| 4:467 | text | diagram/data/pseudocode; not an executable program |
| 4:473 | text | diagram/data/pseudocode; not an executable program |
| 4:479 | text | diagram/data/pseudocode; not an executable program |
| 4:487 | text | diagram/data/pseudocode; not an executable program |
| 4:494 | text | diagram/data/pseudocode; not an executable program |
| 4:503 | text | diagram/data/pseudocode; not an executable program |
| 4:513 | sh | command recipe; environment/workload dependent |
| 4:521 | text | diagram/data/pseudocode; not an executable program |
| 4:529 | text | diagram/data/pseudocode; not an executable program |
| 4:596 | text | diagram/data/pseudocode; not an executable program |
| 4:638 | text | diagram/data/pseudocode; not an executable program |
| 4:656 | text | diagram/data/pseudocode; not an executable program |
| 4:663 | rust | reference excerpt; covered by workspace, not standalone |
| 4:677 | text | diagram/data/pseudocode; not an executable program |
| 4:699 | sh | command recipe; environment/workload dependent |
| 4:707 | text | diagram/data/pseudocode; not an executable program |
| 4:855 | text | diagram/data/pseudocode; not an executable program |
| 4:922 | text | diagram/data/pseudocode; not an executable program |
| 4:936 | text | diagram/data/pseudocode; not an executable program |
| 4:1094 | text | diagram/data/pseudocode; not an executable program |
| 4:1106 | text | diagram/data/pseudocode; not an executable program |
| 5:27 | text | diagram/data/pseudocode; not an executable program |
| 5:89 | rust | reference excerpt; covered by workspace, not standalone |
| 5:160 | text | diagram/data/pseudocode; not an executable program |
| 5:297 | rust | reference excerpt; covered by workspace, not standalone |
| 5:370 | text | diagram/data/pseudocode; not an executable program |
| 5:380 | text | diagram/data/pseudocode; not an executable program |
| 5:436 | rust | reference excerpt; covered by workspace, not standalone |
| 5:475 | rust | reference excerpt; covered by workspace, not standalone |
| 5:549 | sh | command recipe; environment/workload dependent |
| 5:582 | text | diagram/data/pseudocode; not an executable program |
| 5:701 | text | diagram/data/pseudocode; not an executable program |
| 6:17 | text | diagram/data/pseudocode; not an executable program |
| 6:88 | text | diagram/data/pseudocode; not an executable program |
| 6:135 | rust | reference excerpt; covered by workspace, not standalone |
| 6:213 | text | diagram/data/pseudocode; not an executable program |
| 6:242 | text | diagram/data/pseudocode; not an executable program |
| 6:346 | rust | reference excerpt; covered by workspace, not standalone |
| 6:510 | text | diagram/data/pseudocode; not an executable program |
| 6:528 | rust | reference excerpt; covered by workspace, not standalone |
| 6:674 | text | diagram/data/pseudocode; not an executable program |
| 6:738 | rust | reference excerpt; covered by workspace, not standalone |
| 6:969 | rust | reference excerpt; covered by workspace, not standalone |
| 6:1012 | rust | reference excerpt; covered by workspace, not standalone |
| 7:40 | text | diagram/data/pseudocode; not an executable program |
| 7:107 | text | diagram/data/pseudocode; not an executable program |
| 7:166 | rust | reference excerpt; covered by workspace, not standalone |
| 7:197 | text | diagram/data/pseudocode; not an executable program |
| 7:250 | text | diagram/data/pseudocode; not an executable program |
| 7:287 | text | diagram/data/pseudocode; not an executable program |
| 7:474 | text | diagram/data/pseudocode; not an executable program |
| 7:496 | rust | reference excerpt; covered by workspace, not standalone |
| 7:847 | text | diagram/data/pseudocode; not an executable program |
| 7:968 | text | diagram/data/pseudocode; not an executable program |
| 7:990 | text | diagram/data/pseudocode; not an executable program |
| 7:1004 | text | diagram/data/pseudocode; not an executable program |
