#Generate a left canonical MPS from a random MPS
using TensorOperations
#MPS_psi=; # input tensor array here
MPS_left_psi=Array{Complex128,3}[];

L=length(MPS_left_psi); # Length of the physical chain
d=size(MPS_psi[1])[1]; # size of the local Hilbert space

#Treat first site separately
M1=squeeze(MPS_psi[1],2); # first bond index is 1, so remove that
M2=MPS_psi[2];
sz_M2=size(M2);
M1_SVD=svdfact(M1);
sz_U=size(M1_SVD[:U]);
A1=zeros(Complex128,sz_U[1],1,sz_U[2]);
A1[:,1,:]=M1_SVD[:U];
push!(MPS_left_psi,A1);
sz_U=size(M1_SVD[:U]);
M_SV=(diagm(M1_SVD[:S])*M1_SVD[:V]');
M2_tilde=zeros(Complex128,d,sz_U[2],sz_M2[3]);
@tensor M2_tilde[s1,a1,a2]=M_SV[a1,a3]*M2[s1,a3,a2];

for p=3:length(MPS_psi)
	#each loop makes an SVD of M2_tilde, pushes U to MPS_left_psi and multiplies SV' to M3
	M3=MPS_psi[p];
 	sz_M3=size(M3);
        sz_M2_tilde=size(M2_tilde);
 	M2_tilde_mat=reshape(M2_tilde[:],d*sz_M2_tilde[2],sz_M2_tilde[3]);
        M2_tilde_SVD=svdfact(M2_tilde_mat);
        sz_U=size(M2_tilde_SVD[:U]);
        A2=zeros(Complex128,d,sz_M2_tilde[2],sz_U[2]);
        A2=reshape(M2_tilde_SVD[:U],d,sz_M2_tilde[2],sz_U[2]);
	push!(MPS_left_psi,A2);
        M2_tilde_SV=(diagm(M2_tilde_SVD[:S])*M2_tilde_SVD[:V]');
	M2_tilde=zeros(Complex128,d,sz_U[2],sz_M3[3]);#define new M2_tilde for next multiplication
  	@tensor M2_tilde[s1,a1,a2]=M2_tilde_SV[a1,a3]*M3[s1,a3,a2];
end
push!(MPS_left_psi,M2_tilde);#push last tensor. This might be garbage. What to do with this last tensor


# check left normalization
p=2
W=MPS_left_psi[p]
W2=zeros(Complex128,size(W)[3],size(W)[3]);
@tensor W2[a1,a3]=conj(W[s1,a2,a1])*W[s1,a2,a3];
W2

