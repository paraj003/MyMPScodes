#module to define the U, Ubar tensors for the Floquet Hubbard Hamiltonian, (See Schoolwock, and FloquetMott/Notes/systemnotes.lyx)
#consists of two functions:
#g(): Step function for the time dependence
#U_Ubar(): Takes H(t) on two sites (i,i+1) and outputs U and Ubar


module FloquetHubbard_timeevolution
using TensorOperations
export g, U_UBar

function g(t,T)
        if rem(t,T)<T/2
                return -1.0
        else
                return 1.0
        end
end

function U_Ubar(T,t,dt,d,V0)
#Takes Hamiltonian, and outputs a tuple, with tensors U and Ubar for evolution.
#inputs: T: Time period, t:current time, dt: time of evolution, d:local Hilbertspace dimension,V0: interaction potentian
        #define the two-site Hamiltonian
        H2site=[0.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 V0*g(t,T)]
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
end

end

