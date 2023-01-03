# DP Matrix Factorization for Streaming Linear Operators

This directory contains the code for improving expected reconstruction error for
differentially private linear operators via the matrix mechanism.

The code here implements both gradient and fixed-point based algorithms to
compute optimal factorizations, and integrates these factorizations with
federated learning for the purpose of training machine learning models.

Link to paper forthcoming. We note that the code in this directory generally
uses the terms S = WH to refer to a factorization of a matrix, whereas the paper
usually represents such a factorization as A = BC, with S reserved for a special
(prefix sum) matrix.
