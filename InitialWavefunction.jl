#File containing different initialization of the initial wave function, in MPS form.,


####Random initialization
d=2;
Psi1=rand(d^L,1)+im*rand(d^L,1);
Psi_in=Psi1/sqrt(sum(abs(Psi1).^2)); 


## Eigenstate taken from ED data
#Test1 : Load Eigenstate and do evolution
#N site solution, V0=8.111308307896872
filenumber=63
N=12
filename1="/home/paraj/Documents/MyMPScodes/textdata/TestData_N_$N\_$filenumber\_States_real.txt"
filename2="/home/paraj/Documents/MyMPScodes/textdata/TestData_N_$N\_$filenumber\_States_imag.txt"
filename3="/home/paraj/Documents/MyMPScodes/textdata/TestData_N_$N\_$filenumber\_Spectrum.txt"
filename4="/home/paraj/Documents/MyMPScodes/textdata/TestData_N_$N\_$filenumber\_UT_real.txt"
filename5="/home/paraj/Documents/MyMPScodes/textdata/TestData_N_$N\_$filenumber\_UT_imag.txt"
eigenvector_real=readdlm(filename1);
eigenvector_imag=readdlm(filename2);
eigenvalues=readdlm(filename3);
UT_real=readdlm(filename4);
UT_imag=readdlm(filename5);
eigenvectors=eigenvector_real+im*eigenvector_imag;
floquetUnitary=UT_real+im*UT_imag;

#choose wavefunction
choice=1;
@show eigenvalues[choice]
Psi_in=eigenvectors[:,choice];

