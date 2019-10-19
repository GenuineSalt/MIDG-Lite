MIDG-Lite
====

Mini Discontinuous Galerkin Maxwells Time-domain Solver Optimised for the CPU with BLAS

Notes:  
MIDG-Lite is a lightweight version of MIDG2 (see license for information about the author of MIDG2).  
The OCCA layer has been removed and replaced with simpler kernels written in C.  
The kernels perform matrix multiplications using the BLAS library.   
MIDG-Lite was developed for an undergraduate honours thesis that is about optimising applications for the CPU.  
While MIDG2 works well on the GPU it could be improved for the CPU.  

MIDG-Lite is not fully optimised.   
Further improvements can be made by:
1. Replacing more vector operations with BLAS calls
2. Restructuring the data to increase compatibility with BLAS functions (remove overhead cost of memory transfers)
3. Restructuring the data to better fit the CPU cache
