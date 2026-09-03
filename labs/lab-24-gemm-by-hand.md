# Lab 24 — GEMM by Hand

**Chapter:** 6. **Level:** BUILD.

## Prerequisites

Labs 22–23 and matrix shape notation.

## Build

Multiply

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
B=\begin{bmatrix}7&8\\9&10\\11&12\end{bmatrix}.
$$

Write the dot product for every one of the four output cells before running
`matmul_reference`.

## Oracle

The exact output is

$$
AB=\begin{bmatrix}58&64\\139&154\end{bmatrix}.
$$

Compare the hand work, Python oracle, and Rust result including shape `[2,2]`.

## Break / prove

Transpose only `B` and explain the shape failure. Then transpose `A` as a view,
choose a compatible right matrix, and prove the reference path accepts its
non-canonical strides.

## Extend

Trace one output element from logical indices through Chapter 5's stride
equation to physical offsets.

## Cleanup

Keep the full-vector assertion rather than checking only one convenient cell.
