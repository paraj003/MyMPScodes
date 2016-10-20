#Takes a generic quantum state written as a vector and rewrites it as an MPS in left canonical form.

#input is a column vector of dimension (d^L,1) where d is the size of the local Hilbert space.

using TensorOperations

#definitions CHECK!
d=2; #Size of local hilbert space
L=10; # number of sites
Psi_in=rand(d^L,1); #column vector, arranged in binary order (for d=2) 
MPS_left=Array{Float64,3}[] #defines an array of rank 3 tensors (for MPS)
#initialize leftcanon as a tensor
r_arr=fill(0,L-1);#array of ranks of the tensors only L-1 because first and last tensors have only 1 index each

Psi1=Psi_in/sqrt(sum(Psi_in.^2)); #normalized eigenvector(not necessary geneerally)

########First do site 1.
Psimod=reshape(Psi1,d^(L-1),d^(1));# the two columns correspond to two different cases of the first site.
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
 push!(MPS_left,A_tensor)
 SV=(diagm(PsiSVD[:S])*(PsiSVD[:V]')).'[:]; # r[p] X d^(L-p) array
end
#A_tensor on site L is nothing but SV
A_tensor=reshape(SV,d,r_arr[L-1],1);
push!(MPS_left,A_tensor);


#contract all  bond indices to check the tensors for 3 sites
Psi=MPS_left[1];
for p=1:L-1
 A=MPS_left[1];
 B=MPS_left[2];
 D=MPS_left[3];
 Psi=zeros(2,2,2);
 @tensor begin
  Psi[s1,s2,s3]=A[s1,1,a1]*B[s2,a1,a2]*D[s3,a2,1] 
 end
end

