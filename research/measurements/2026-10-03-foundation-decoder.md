# Chapters 9--11 correctness depth pass — 2026-10-03

## Scope and reproduction

Standard-library Python 3.9.6 on macOS. Illustrative untrained weights;
no accelerator, external framework, downloaded model or timing experiment.

```bash
python3 code/foundations/decoder_trace.py
python3 -m unittest discover -s code/foundations \
  -p 'test_decoder_trace.py' -v
python3 -m unittest discover -s code/foundations -p 'test_*.py'
```

Seventeen new tests passed; the complete foundation suite passed 80 tests.
No measured latency, throughput, hardware allocation or language quality is
claimed. Derived FLOPs, packed payloads and observed call counts retain their
explicit boundaries.

## Source identity

SHA-256 of the files used for the recorded run:

| File | SHA-256 |
| --- | --- |
| `code/foundations/decoder_trace.py` | `102060bb60200170d272d276190495302d4fb06ebe63ab82c0ac080213f0b0c2` |
| `code/foundations/test_decoder_trace.py` | `e74d21da1fc3b058953191ab5a4a112ce65eb82a86c13f66e41ca8a3178503d4` |
| `code/foundations/model.py` | `4c235c635f8e332d5c9e211a2b642ef2e995b14aa3ace02d24e7634f6ac474f3` |

## Oracles, coverage and negative controls

- Offset mask: prefix three plus chunk two permits four/five keys.
  Equal-score values `0,1,2,3,4` give means 1.5 and 2.
  The deliberately wrong upper-left triangle gives 0 and 0.5.
- Three asymmetric attention fixtures at widths two/four/six use a separate
  70-digit Decimal dot/scale/exp/reduction oracle, with no model primitives.
  Uniform means, convex coordinate bounds and value translation are separate
  properties at the pre-output-projection boundary.
- Residual fixture: scalar equations independently derive normalized inputs,
  gate products and final `[5.523036211784106,3.1327092572515194]`.
  Attention update is supplied, not calculated from Q/K/V.
  Swapping gate/up branches changes the tested function.
- Hidden tiling: twenty shape/tile cases compare grouped scalar sums with
  materialized SwiGLU. Uneven hidden width five forces tail handling.
  This comparison shares primitives and is not an independent activation
  oracle. Reported scratch is products-list length, not process memory.
- Decoder parity: 125 three-token histories in each of three head/layer
  configurations (375 histories, 1,125 prefix comparisons), plus the existing
  longer-prefix/generation tests. Shared primitives remain shared.
- Independent assembly: zero attention-output/down matrices leave a chosen
  embedding unchanged; final normalization/output must equal
  `[3,4,-1,0,0]/sqrt(25/4+1e-5)`. This does not test nonzero sublayer updates.
- Position-reset negative control: first token agrees; resetting position
  before the second causes maximum logit discrepancy
  `0.0011518901159686679` (greater than `1e-5`).
  Separate caches and separate layer values are checked.
- Instrumentation observes 24 projected rows / 124 attention pairs for full
  lengths three/four/five, and ten rows / 60 pairs for five cached appends.
  Both count two layers and two query heads, without timing.
- Four query heads map to two distinct KV heads in consecutive groups.
  Parameter enumeration confirms 120 entries per layer, 280 overall;
  matched two-matrix/gated FFN examples both have 144 entries.
- Budget/end stop precedes final feedback; four selections leave six
  evaluated positions. Invalid budgets/end IDs are rejected within the
  trace's declared interface. Generic cache rollback is not implemented.

All floating comparisons use `1e-12` absolute tolerance on bounded fixtures,
except the previously printed six-decimal regression row. Zero errors in
this trace do not promise bitwise agreement across optimized kernels.
The original teaching decoder has not been modified.

## Raw trace output

```json
{
  "chunk": [
    {
      "position": 3,
      "permitted": [
        0,
        1,
        2,
        3
      ],
      "probabilities": [
        0.25,
        0.25,
        0.25,
        0.25
      ],
      "output": [
        1.5
      ]
    },
    {
      "position": 4,
      "permitted": [
        0,
        1,
        2,
        3,
        4
      ],
      "probabilities": [
        0.2,
        0.2,
        0.2,
        0.2,
        0.2
      ],
      "output": [
        2.0
      ]
    }
  ],
  "residual": {
    "x": [
      3.0,
      4.0
    ],
    "attention_input": [
      0.8320502943378437,
      1.1094003924504583
    ],
    "attention_update": [
      1.0,
      -2.0
    ],
    "r": [
      4.0,
      2.0
    ],
    "ffn_input": [
      1.2344267996967353,
      0.6172133998483676
    ],
    "gate": [
      1.2344267996967353,
      0.6172133998483676
    ],
    "up": [
      1.8516401995451028,
      0.6172133998483676
    ],
    "product": [
      1.770496979357084,
      0.2474607675729773
    ],
    "ffn_update": [
      1.5230362117841065,
      1.1327092572515194
    ],
    "next": [
      5.523036211784106,
      3.1327092572515194
    ]
  },
  "layer_zero_position_two": {
    "layer": 0,
    "position": 2,
    "residual": [
      0.07934811462612273,
      -0.19227949837591138,
      -0.02979980516283976,
      0.19995858002853384
    ],
    "q": [
      [
        0.32225334322955995,
        0.09175466646999944
      ],
      [
        0.7478374223259073,
        -0.2353678943411222
      ]
    ],
    "k": [
      [
        0.4430894461905178,
        -0.3326320489292841
      ]
    ],
    "v": [
      [
        0.026981642600238432,
        0.31139186310286215
      ]
    ]
  },
  "matrix_parameters": 280,
  "layer_matrix_parameters": 120,
  "last_logits": [
    0.09392682604603625,
    -0.20483678530459576,
    -0.45009600019633084,
    -0.5777878806697748,
    -0.5545586666213511
  ],
  "generation": {
    "events": [
      {
        "evaluated": 3,
        "selected": 0,
        "max_logit_error": 0.0,
        "logits": [
          0.09392682604603625,
          -0.20483678530459576,
          -0.45009600019633084,
          -0.5777878806697748,
          -0.5545586666213511
        ]
      },
      {
        "evaluated": 4,
        "selected": 4,
        "max_logit_error": 0.0,
        "logits": [
          -0.3264708201729855,
          -0.5164369877207347,
          -0.5715072219234747,
          -0.47729690119340984,
          -0.2584142341751141
        ]
      },
      {
        "evaluated": 5,
        "selected": 0,
        "max_logit_error": 0.0,
        "logits": [
          0.5221681713576652,
          0.4338666269787261,
          0.23223694192602,
          -0.03005419802472352,
          -0.284495030600361
        ]
      },
      {
        "evaluated": 6,
        "selected": 4,
        "max_logit_error": 0.0,
        "logits": [
          -0.3178828211278452,
          -0.5162520583428302,
          -0.5797736666945308,
          -0.491855483247457,
          -0.27546217880573887
        ]
      }
    ],
    "cache_position": 6,
    "cache_layer_lengths": [
      6,
      6
    ]
  }
}

```
