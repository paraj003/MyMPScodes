# MyMPScodes
This is a list of all MPS codes.

leftcanon.jl : Given a column vector that is the wave function, this file converts into an MPS states, which is an array of tensors A[s1,a1,a2], s1 being the physical dimension and a1, a2 being the bond dimensions. It exactly follows, Schoolwock review section 4.1.3. The MPS tensors are not compressed in this case.

contraction.jl : Takes two MPS wavefunctions |psi> and |phi> and computes <phi|psi>
