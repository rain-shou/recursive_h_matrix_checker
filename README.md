# Recursive H-Matrix Checker -- Experimental Code

This repository contains the reference implementations and benchmark programs used in the bachelor thesis:

"Implementation and Performance Optimization of a Recursive H-Matrix Judging Algorithm"

## Repository Structure

- `original`  
  Reference implementaions of the original recursive algorithm proposed by Li, as well as LU and Gauss-based methods.  

- `blas2/`  
  BLAS-2 based implementations of the recursive algorithm.
  Seperate directories are provided for single-threaded and multi-threaded executions.  

- `block_blas3/`  
  Blocked BLAS-3 based implementations proposed in this thesis.  
  Both single-threaded and multi-threaded versions are inclueded.

## Notes
- This code is intended for research and benchmarking pruposes.
- Expreimental results reported in the thesis were obtained using this code.
- Detailed explanations of the algorithms and performance analysis can be found in the thesis.
