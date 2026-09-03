# Lab 23 — GEMV by Hand

**Chapter:** 6. **Level:** CHECK.

## Prerequisites

Lab 22 and the `[M,K] × [K] -> [M]` GEMV contract.

## Build

Use

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
x=\begin{bmatrix}2\\-1\\0.5\end{bmatrix}.
$$

Calculate each row dot product, then run `gemv_reference` on the same values.

## Oracle

The expected owned output has shape `[2]` and values `[1.5, 6.0]`. Confirm the
same vector in the Chapter 6 Python oracle.

## Break / prove

Change the vector shape to `[2]` while leaving the matrix `[2,3]`. Require a
typed `InnerDimensionMismatch`. Then pass a rank-2 vector and require a rank
error before numerical work.

## Extend

Store useful matrix and vector values with padding and create strided views.
Prove GEMV follows logical indices, not adjacent physical positions.

## Cleanup

Retain the asymmetric fixture because square, all-positive examples hide
orientation errors.
