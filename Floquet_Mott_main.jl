#Main file for running Floquet Mott tMPS simulations

using FloquetHubbard_timeevolution
using MPS_module
using TensorOperations

d=2;  #Size of local hilbert space
L=12; # number of sites
D=300; #cutoff bond-dimension
T=2*pi/1.7;  #Time period 
V0=8.111308307896872; #Interaction strength
Ndiv=100; #Number of divisions of T
dt=T/Ndiv; #time-step, take an integer divison of T
NT=1; #number of timeperiods

########## Input wavefunction
#### take Psi_in from InitialWavefunction.jl

Psi0=Wavefunc_to_MPS_left(Psi_in,d,L); #generate MPS from the wavefunction
Psi0_compress=Compress_SVD_MPS_left(Psi0,D);




#Psi0=#Initial MPS state here,in left canonical form|ψ(0)>
Psi_t=Array{Complex128,3}[] #empty array of rank 3 tensors for the final time-evolved wavefunction |ψ(t)>

#Generate Tensors for time evolution:
#Odd bonds: U-Ubar-U-Ubar......U-Ubar : evolve for dt/2
#even bonds: I-U-Ubar-U-Ubar.....U-Ubar-I : evolve for dt

#list of U and Ubars for different times in a single time period
U_t_array_even=Array{Complex128,3}[]
Ubar_t_array_even=Array{Complex128,3}[]
U_t_array_odd=Array{Complex128,3}[]
Ubar_t_array_odd=Array{Complex128,3}[]
for p=1:Ndiv
        t=p*dt
	(U,Ubar)=U_Ubar(T,t,dt,d,V0)
        (U1,U1bar)=U_Ubar(T,t,dt/2,d,V0)
        push!(U_t_array_even,U);
        push!(Ubar_t_array_even,Ubar);
        push!(U_t_array_odd,U1);
        push!(Ubar_t_array_odd,U1bar);
end

#####Trotter decomposition: odd-even-odd
Psi_temp=copy(Psi0_compress); #Evolve this
quasienergy=zeros(Complex{Float64},NT); #finds quasienergy as a function of time.
for n=1:NT
	@show n
	for p=1:Ndiv
                @show p
		U_even=copy(U_t_array_even[p])
		Ubar_even=copy(Ubar_t_array_even[p])
	        U_odd=copy(U_t_array_odd[p])
	        Ubar_odd=copy(Ubar_t_array_odd[p])
		#First trotter step:odd bonds
		A0_SV=ones(Complex128,1,1); #(Start on left most site)
	 	for l1=1:div(L,2)
			A1=copy(Psi_temp[2*l1-1]);
			A2=copy(Psi_temp[2*l1]);
			(A1out,A2out,A2_SV)=Apply_U_Ubar_left(A0_SV,A1,A2,U_odd,Ubar_odd,D)
			A0_SV=copy(A2_SV)
			Psi_temp[2*l1-1]=copy(A1out)
			Psi_temp[2*l1]=copy(A2out)
		end
		#@show A0_SV
		#Normalization of the state. Remember the applicaiton of U_Ubar is not normalized. So have to divide by a numerical factor. This is just the S eigenvalue for the SVD of the last tensor. So divide by abs|A2_SV|.
		Psi_temp[L]=copy(Psi_temp[L]/sum(sqrt(abs(A0_SV))))
		#Second Trotter step : even bonds
		A0_SV=eye(Complex128,size(Psi_temp[2])[2]); #(Start on left most site)
		#@show abs(Contract_MPS_left(Psi0_compress,Psi_temp))
                for l1=1:div(L,2)-1
                        A1=copy(Psi_temp[2*l1]);
                        A2=copy(Psi_temp[2*l1+1]);
                        (A1out,A2out,A2_SV)=Apply_U_Ubar_left(A0_SV,A1,A2,U_even,Ubar_even,D)
                        A0_SV=copy(A2_SV)
                        Psi_temp[2*l1]=copy(A1out)
                        Psi_temp[2*l1+1]=copy(A2out)
                end
                
		#Multiply A0_SV to the last tensor
		A3=Psi_temp[L];
		A4=zeros(Complex128,size(Psi_temp[L]));
		@tensor A4[s1,a1,a2]=A0_SV[a1,a3]*A3[s1,a3,a2]
		Psi_temp[L]=copy(A4)
		#@show abs(Contract_MPS_left(Psi0_compress,Psi_temp))
		#Third Trotter step :odd bonds again


                A0_SV=ones(Complex128,1,1); #(Start on left most site)
                for l1=1:div(L,2)
                        A1=copy(Psi_temp[2*l1-1]);
                        A2=copy(Psi_temp[2*l1]);
                        (A1out,A2out,A2_SV)=Apply_U_Ubar_left(A0_SV,A1,A2,U_odd,Ubar_odd,D)
                        A0_SV=copy(A2_SV)
                        Psi_temp[2*l1-1]=copy(A1out)
                        Psi_temp[2*l1]=copy(A2out)
                end
                Psi_temp[L]=copy(Psi_temp[L]/sum(sqrt(abs(A0_SV))))
		#@show abs(Contract_MPS_left(Psi0_compress,Psi_temp))
	end
	#Psi_t=copy(Psi_temp);
	#Calculate some observable here.
	quasienergy[n]=im/(n*T)*(log(Contract_MPS_left(Psi0_compress,Psi_temp)))
	@show(quasienergy,eigenvalues[choice])
end


##Code for testing output :




