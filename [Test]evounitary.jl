#generate the MPO for the time evolution step for two sites.
#H_MPS frim systemNotes in FloquetMott 
#write down the evolution operator for sites i and i+1, exp(-iH_{i,i+1} dt)

using TensorOperations

T=1; #timeperiod 
t=1;  # time for the evolution 
dt=T/10; #time step
L=4 #no. of sites
d=2; #physical Hilbert space dimesnsion
V0=1;#Interaction potential
D=30; #bond dimension cutoff

#function takes in time and outputs step function with period T(modify to modify time dependence)
function g(t,T)
	if rem(t,T)<T/2
		return -1.0
	else 
		return 1.0
	end
end

#Takes Hamiltonian, and outputs a tuple, with tensors U and Ubar for evolution.
function U_Ubar(T,t,dt,d,V0)
	#define the two-site Hamiltonian
	H2site=[0.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 V0*g(t)]
	U2site=expm(-1im*H2site*dt);
	#define P from section 7.1.2 Schoolwock,section 8 on systemNotes
	P=zeros(Complex128,4,4);
	p=[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4];
	q=[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4];
	s1arr=[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]+1;#add 1 to make them correspond to physical index
	s2arr=[0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]+1;
	s1primearr=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]+1;
	s2primearr=[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]+1;
	ptilde=[1,1,2,2,1,1,2,2,3,3,4,4,3,3,4,4];
	qtilde=[1,2,1,2,3,4,3,4,1,2,1,2,3,4,3,4];
	for p1=1:16
		P[p[p1],q[p1]]=U2site[ptilde[p1],qtilde[p1]];
	end
	O_SVD=svdfact(P);
	O_U1=O_SVD[:U]*diagm(sqrt(O_SVD[:S]));
	O_Ubar1=diagm(sqrt(O_SVD[:S]))*(O_SVD[:V])';
	U=zeros(Complex128,d,d,length(O_SVD[:S]))#have removed the index of dim 1 (sec. 7.1.2)
	Ubar=zeros(Complex128,d,d,length(O_SVD[:S]))#I have removed the index of dimension 1
	for p1=1:16
		U[s1arr[p1],s1primearr[p1],:]=O_U1[ptilde[p1],:]
        	Ubar[s2arr[p1],s2primearr[p1],:]=O_Ubar1[:,qtilde[p1]]
	end
	return U,Ubar
end;

## apply U and Ubar tensor on two MPS tensors, (A1)-(A2)=> compress by SVD

A1_tensor= # Tensor acting on s1prime
A2_tensor= # Tensor acting on s2prime 
A0_SV=ones(Complex128,1,1) #inherited SV' from the previous site. is 1 on the left most site.


function Apply_U_Ubar(A0_SV,A1_tensor,A2_tensor,U,Ubar,D)

	A1_tilde_tensor=zeros(Complex128,d,size(A1_tensor)[2],size(A1_tensor)[3],d^2);
	A2_tilde_tensor=zeros(Complex128,d,size(A2_tensor)[2],d^2,size(A2_tensor)[3]);

	@tensor A1_tilde_tensor[s1,a1,a2,k]=A0_SV[a1,a3]*A1_tensor[s1prime,a3,a2]*U[s1,s1prime,k];
	@tensor A2_tilde_tensor[s2,a1,k,a2]=A2_tensor[s2prime,a1,a2]*Ubar[s2,s2prime,k];

	#Now group indices together : a2 and k for A1_tilde_tensor, and (a1,k) for A2_tilde_tensor.
	# then compress by SVD reduce Dd^2->D 
	A1_out_tensor=reshape(A1_tilde_tensor,d,size(A1_tensor)[2],size(A1_tensor)[3]*d^2);
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



~ 
