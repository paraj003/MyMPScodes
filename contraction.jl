#Given two MPS states, MPS_left_phi (in left-canonical form), and MPS_left_psi calculate the contraction. <phi|psi>
using TensorOperations


MPS_left_psi=MPS_left; # input tensor array here. Tensor is always stored as M(s1,a1,a2) where s1 is the physical index, a1 and a2 are the bond indices.
MPS_left_phi=MPS_left; # 2nd input tensor array here.
MPS_left_phi_dag=Array{Complex128,3}[]; #Empty array for phi^†
@show typeof(MPS_left_phi_dag)

L=length(MPS_left_psi); # Length of the physical chain
d=size(MPS_left_psi[1])[1]; # size of the local Hilbert space

#generate conjugate of phi
for i=1:L
	M_tilde=MPS_left_phi[i]
	sz1=size(M_tilde)
	M_tilde_dag=zeros(Complex128,sz1[1],sz1[3],sz1[2]);
	@tensor M_tilde_dag[s1,a1,a2]=conj(M_tilde[s1,a2,a1]);
	push!(MPS_left_phi_dag,M_tilde_dag)
end

#start from left, contract-bond type (1) (See Fig 21 Schoolwock) :
#the dimension of matrices are (d,a1,1)*(d,1,a2) summed over the physical index
M_tilde_dag=squeeze(MPS_left_phi_dag[1],3);#remove the unit-dimensions
M=squeeze(MPS_left_psi[1],2);
sz_M_tilde_dag=size(M_tilde_dag); 
sz_M=size(M);
M1=zeros(Complex128,sz_M_tilde_dag[2],sz_M[2]);#new tensor with physical index summed
@tensor M1[a1,a2]=M_tilde_dag[s1,a1]*M[s1,a2];

for p=2:length(MPS_left_psi)
	#contract bond of type (2) (Fig 21)
	#dimension of matrices : (1,a1,a2)*(d,a1,a3)
	sz_M1=size(M1);
	M_tilde_dag=MPS_left_phi_dag[p];
	sz_M_tilde_dag=size(M_tilde_dag);
	M2=zeros(Complex128,d,sz_M_tilde_dag[2],sz_M1[1]);
	@tensor M2[s1,a1,a3]=M_tilde_dag[s1,a1,a2]*M1[a2,a3];

	#contract bond of type (3) and (4) simultaneously to generate a matrix like M1
	sz_M2=size(M2);
	M=MPS_left_psi[p];
	sz_M=size(M);
	M1=zeros(Complex128,sz_M2[2],sz_M[3]);
	@tensor M1[a1,a2]=M2[s1,a1,a3]*M[s1,a3,a2];
end
phi_psi_innerprod=M1;
