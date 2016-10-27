#Generate a left canonical MPS from a random MPS
using TensorOperations
MPS_psi=; # input tensor array here
MPS_left_psi=Array{Complex128,3}[];

L=length(MPS_psi); # Length of the physical chain
d=size(MPS_psi[1])[1]; # size of the local Hilbert space
D=5; #define cutoff for the number of bond-dimensions to keep.


#Treat first site separately
M1=squeeze(MPS_psi[1],2); # first bond index is 1, so remove that
M2=MPS_psi[2];
sz_M2=size(M2);
M1_SVD=svdfact(M1);
sz_U=size(M1_SVD[:U]);
BD=min(sz_U[2],D); #Finds the appropriate bonddimension, BD : number of columns is chi or that in U depending on whichever is smaller
A1=zeros(Complex128,sz_U[1],1,BD);
A1[:,1,:]=M1_SVD[:U][:,1:BD];
push!(MPS_left_psi,A1);
M_S=M1_SVD[:S][1:BD];
M_SV=(diagm(M1_SVD[:S][1:BD])*(M1_SVD[:V][:,1:BD])');
M2_tilde=zeros(Complex128,d,BD,sz_M2[3]);
@tensor M2_tilde[s1,a1,a2]=M_SV[a1,a3]*M2[s1,a3,a2];

for p=3:length(MPS_psi)
	#each loop makes an SVD of M2_tilde, pushes U to MPS_left_psi and multiplies SV' to M3
	M3=MPS_psi[p];
 	sz_M3=size(M3);
        sz_M2_tilde=size(M2_tilde);
 	M2_tilde_mat=reshape(M2_tilde[:],d*sz_M2_tilde[2],sz_M2_tilde[3]);
        M2_tilde_SVD=svdfact(M2_tilde_mat);
        sz_U=size(M2_tilde_SVD[:U]);
        BD=min(sz_U[2],D);
        A2=zeros(Complex128,d,sz_M2_tilde[2],BD);
        A2=reshape(M2_tilde_SVD[:U][:,1:BD],d,sz_M2_tilde[2],BD);
	push!(MPS_left_psi,A2);
        M2_tilde_SV=(diagm(M2_tilde_SVD[:S][1:BD])*((M2_tilde_SVD[:V][:,1:BD])'));
	M2_tilde=zeros(Complex128,d,BD,sz_M3[3]);#define new M2_tilde for next multiplication
  	@tensor M2_tilde[s1,a1,a2]=M2_tilde_SV[a1,a3]*M3[s1,a3,a2];
end
#Normalize the last tensor,
A2=M2_tilde/sqrt(sum(abs(M2_tilde[:].^2)));# the last tensor has d column vectors.Needs to be normalized
push!(MPS_left_psi,A2);#push last tensor. 


# check left normalization
#p=1
#W=MPS_left_psi[p];
#size(W)
#W2=zeros(Complex128,size(W)[3],size(W)[3]);
#@tensor W2[a1,a3]=conj(W[s1,a2,a1])*W[s1,a2,a3];
#W2

