# Lab 12 — Top-k Versus Top-p

Chapter: 4. Artifact: two candidate sets from one distribution.

Use `[A=.40,B=.30,C=.15,D=.10,E=.05]`.

## CHECK

Calculate the survivors for `top_k=2` and for `top_p=.80`. Include the token
that crosses the nucleus threshold.

## BUILD

Encode the probabilities as logits with `ln(p)` and run the Python oracle.
Verify top-k retains A/B while top-p retains A/B/C and renormalizes their mass.

## BREAK

Try `k=0`, `k>V`, `p=0`, `p=1`, and `p>1`. Record which are invalid and which
are deliberate no-ops under ENGINE-1's contract.

## EXTEND

Combine top-k and top-p. Apply ENGINE-1's documented order, then reverse it by
hand. Find a vector where the surviving set changes and explain why order is
part of inference behavior.

