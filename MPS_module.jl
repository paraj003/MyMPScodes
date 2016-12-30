#A module containing all the major functions for MPS operations.
#List of functions :


#Wavefunc_to_MPS():Converts a Wavefunction to MPS 
#Compress_SVD_MPS_left() : Takes an MPS, and compresses it using SVD. Normalization is implemented on the last tensor.
#Contract_MPS_left() : Contracts two MPS given in left canonical form
#[NOT DONE]Apply_MPO()  :Apply generic single-site MPO on a site
# Apply_U_Ubar_left() :For evolution of MPS (A1)-(A2) using U and Ubar, as in section 7.1.2 of Schoolwock and gives a compressed output. The state is not normalized, and the normalization must be done at the end of the tensor.
module MPS_module
using TensorOperations
export Apply_U_Ubar_left, Wavefunc_to_MPS_left,Compress_SVD_MPS_left,Contract_MPS_left,Apply_U_Ubar_left_uncompressed


function Wavefunc_to_MPS_left(Psi_in,d,L)
# An exact representation of a wavefunction in MPS language.
#The wavefunction is entered in lexicographic order, has dimensions (d^L,1).
#
        MPS_left=Array{Complex{Float64},3}[]; #defines an array of rank 3 tensors (for MPS),initialize leftcanon as a tensor
	r_arr=fill(0,L-1);#array of zeros to store bond dimension of the tensors only L-1 because L-1 bonds.

	########First do site 1.
	Psimod=reshape(Psi_in,d^(L-1),d^(1));# the two columns correspond to two different cases of the first site.
	PsiSVD=svdfact(Psimod.'); # need to transpose it, so that each row is particular spin of first site.
	r_arr[1]=length(PsiSVD[:S]); #r[1] number of elements for index a1
	A_tensor=reshape(PsiSVD[:U],d,1,r_arr[1]); #Left most tensor U_sq
	push!(MPS_left,A_tensor);
	SV=(diagm(PsiSVD[:S])*(PsiSVD[:V]')).'[:]; #SV is a matrix r1 X d^(L-1) transposed and reshaped into an array.

	#########iterate sites 2 through L-1
	for p=2:L-1
		Psimod=reshape(SV,d^(L-p),r_arr[p-1]*d);
		PsiSVD=svdfact(Psimod.');
		r_arr[p]=length(PsiSVD[:S]);
		A_tensor=reshape(PsiSVD[:U],d,r_arr[p-1],r_arr[p]);
		push!(MPS_left,A_tensor);
		SV=(diagm(PsiSVD[:S])*(PsiSVD[:V]')).'[:]; # r[p] X d^(L-p) array
	end
	#A_tensor on site L is nothing but SV
	A_tensor=reshape(SV,d,r_arr[L-1],1);
	push!(MPS_left,A_tensor);
	return MPS_left

end

function Compress_SVD_MPS_left(MPS_psi,D)
#Takes input MPS_psi, and converts it into left canonical MPS_left_psi. At the same time, compressing it, by truncating bond dimensions at D.
#Input is an Array of tensors and the cutoff dimension, returns an array of compressed tensors.
#normalization done at the end.
	MPS_left_psi=Array{Complex128,3}[];
	L=length(MPS_psi); # Length of the physical chain
	d=size(MPS_psi[1])[1];
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
	return MPS_left_psi
end

function Contract_MPS_left(MPS_left_phi,MPS_left_psi)
#Given two MPS states, MPS_left_phi (in left-canonical form), and MPS_left_psi calculate the contraction. <phi|psi> Section 4.4 Schoolwock, Fig 21
	MPS_left_phi_dag=Array{Complex128,3}[];
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
	        M2=zeros(Complex128,d,sz_M_tilde_dag[2],sz_M1[2]);
	        @tensor M2[s1,a1,a3]=M_tilde_dag[s1,a1,a2]*M1[a2,a3];
	
	        #contract bond of type (3) and (4) simultaneously to generate a matrix like M1
	        sz_M2=size(M2);
	        M=MPS_left_psi[p];
	        sz_M=size(M);
	        M1=zeros(Complex128,sz_M2[2],sz_M[3]);
	        @tensor M1[a1,a2]=M2[s1,a1,a3]*M[s1,a3,a2];
	end
	phi_psi_innerprod=M1;
	return phi_psi_innerprod[1,1]

end


function Apply_U_Ubar_left(A0_SV,A1_tensor,A2_tensor,U,Ubar,D)
# This function is designed to apply U*Ubar to the MPS keeping left-canonical normalization in mind. A0_SV is the remainder from the SVD on the left of A1, and for first site is just a 1X1 matrix with 1.0 as entry. 
#Input type = A0_SV≡ (BD,BD), A1_tensor≡ (d,BD,BD), A2_tensor ≡ (d,BD,BD), U≡ (d,d,d^2), Ubar≡ (d,d,d^2)
# returns to tensors A1_out_trunc tensor=(d,BD,BD), A2_out_trunc_tensor=(d,BD,BD), A2_SV=(BD,BD). Here BD signifies the bond dimensions, always  BD<=D, 
#((((((((((((Truncation IS NOT NORMALIZED!!!!)))))))))))) 

	d=size(Ubar)[1];#the dimension of the physical Hilbert space
        A1_tilde_tensor=zeros(Complex128,d,size(A1_tensor)[2],size(A1_tensor)[3],d^2);
        A2_tilde_tensor=zeros(Complex128,d,size(A2_tensor)[2],d^2,size(A2_tensor)[3]);

        @tensor A1_tilde_tensor[s1,a1,a2,k]=A0_SV[a1,a3]*A1_tensor[s1prime,a3,a2]*U[s1,s1prime,k];
        @tensor A2_tilde_tensor[s2,a1,k,a2]=A2_tensor[s2prime,a1,a2]*Ubar[s2,s2prime,k];

        #Now group indices together : a2 and k for A1_tilde_tensor, and (a1,k) for A2_tilde_tensor.
        # then compress by SVD reduce Dd^2->D 
        A1_out_tensor=reshape(A1_tilde_tensor,d,size(A0_SV)[1],size(A1_tensor)[3]*d^2);
        A2_out_tensor=reshape(A2_tilde_tensor,d,size(A2_tensor)[2]*d^2,size(A2_tensor)[3]);

        ## combine  physical index and a1
        A1_out_temp_tensor=reshape(A1_out_tensor,d*size(A1_tensor)[2],size(A1_tensor)[3]*d^2);
        A1_temp_SVD=svdfact(A1_out_temp_tensor);

        BD=min(D,length(A1_temp_SVD[:S]))#bond dimension
	A1_out_trunc_tensor=reshape(A1_temp_SVD[:U][:,1:BD],d,size(A1_tensor)[2],BD)
        A2_tilde_trunc_tensor=zeros(Complex128,d,BD,size(A2_tensor)[3]);
        A1_SV=diagm(A1_temp_SVD[:S][1:BD])*(A1_temp_SVD[:V][:,1:BD])'
        @tensor A2_tilde_trunc_tensor[s1,a1,a2]=A1_SV[a1,a3]*A2_out_tensor[s1,a3,a2]

        A2_out_temp_tensor=reshape(A2_tilde_trunc_tensor,d*BD,size(A2_tensor)[3]);
        A2_temp_SVD=svdfact(A2_out_temp_tensor);
        A2_out_trunc_tensor=reshape(A2_temp_SVD[:U],d,BD,size(A2_temp_SVD[:U])[2]);
        A2_SV=diagm(A2_temp_SVD[:S])*(A2_temp_SVD[:V])' ## Send A2_SV to the next tensor.
        return A1_out_trunc_tensor,A2_out_trunc_tensor,A2_SV
end

function Apply_U_Ubar_left_uncompressed(A1_tensor,A2_tensor,U,Ubar,D)
# This function is designed to apply U*Ubar and do not compress.

        d=size(Ubar)[1];#the dimension of the physical Hilbert space
        A1_tilde_tensor=zeros(Complex128,d,size(A1_tensor)[2],size(A1_tensor)[3],d^2);
        A2_tilde_tensor=zeros(Complex128,d,size(A2_tensor)[2],d^2,size(A2_tensor)[3]);

        @tensor A1_tilde_tensor[s1,a1,a2,k]=A1_tensor[s1prime,a1,a2]*U[s1,s1prime,k];
        @tensor A2_tilde_tensor[s2,a1,k,a2]=A2_tensor[s2prime,a1,a2]*Ubar[s2,s2prime,k];

        #Now group indices together : a2 and k for A1_tilde_tensor, and (a1,k) for A2_tilde_tensor.
        A1_out_tensor=reshape(A1_tilde_tensor,d,size(A1_tensor)[2],size(A1_tensor)[3]*d^2);
        A2_out_tensor=reshape(A2_tilde_tensor,d,size(A2_tensor)[2]*d^2,size(A2_tensor)[3]);

       return A1_out_tensor,A2_out_tensor
end



end
