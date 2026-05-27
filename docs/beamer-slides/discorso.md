# Exam prep: 27/5

## Introduction
Hello everyone,
In my seminar I will talk about the Harrow-Hassidim-Llloyd algorithm, a quantum algorithm introduced in 2009 to solve linear systems. 

This algorithm handles a particular case of linear system, Ax=b associated to sparse and hermitian matrices. An important thing to make clear is that the output of this algorithm is not a solution of the linear system, but the solution is encoded in the amplitudes of the qubits. For this reason, it is not possible to extract the actual solution x but only some quadratic functions related to some operator M.

The theoretical advantage of this quantum algorithm is that it provides a logarithmic scaling with respect to matrix dimension, beating classical algorihms for hermitian sparse matrices which scale linearly.

## Basic idea
The idea beyond the HHL algorithm is to perform the matrix inversion by inverting the eigenvalues of matrix A. In this way we can compute the solution of the linear system by applying the inverse of A to b. The solution x can be then expressed in the eigenvector basis as noted in the slide.

## HHL
Let's enter in the actual implementation of the algorithm to see how is such eigenvalue inversion actually implemented. Here I reported a sketch of the quantum circuit. 

We got three kind of registers. In the b-register the vector b is encoded using amplituded encoding. For this reason, the number of qubits needed, n_b, is such that N_b matches the system dimension. This register will be firstly used to store the vector b and then after the circuit execution it will store the solution, always encoded using amplitude encoding. 

The c-registers will be used to encode the eigenvalues of A after the quantum phase estimation phase. The number of qubits used in these registers is an hyperparameter of the algorithm. Later, I will show some guidelines to set it. 

Finally the ancilla qubit will be used to perform the eigenvalues inversion. This is a single qubit. If its measurement is one the result in the b-register will be stored, if not it will be discarded.

All the qubits are initialised to ket 0.

Looking at the circuit we can then appreciate the steps of the HHL algorithm. Firstly, b has to be encoded through amplitude encoding. This is done in the state preparation step. Then, in the quantum phase estimation step the eigenvalues of A are encoded in the c-register and are inverted in the ancilla qubit rotation step. After this step the registers are entangled and an inverse quantum phase estimation step is needed to disentagle them and store the solution in the b registers.

Now we will review each step in detail.

## State preparation
As said, the state preparation step encodes b in the b-registers thorugh amplitude encoding. Here I reported very simple example to show what it means to encode (0,1) through the X-gate. Depending on the vector b to be encoded the state preparation can be more or less involved. The thing to recall is that this step is specific to the vector b considered

## Quantum Phase Estimation
The Quantum Phase Estimation step is more intersting. As anticipated, the goal is to estimate the eigenvalues of A but we got two major issues: how do we encode matrix A in the circuit and how do we estimate its parameters. 

The first problem is tackled using Hamiltonian simulation. Let's understand the problem first. We got a matrix A which is non-unitary in general. What we do is defining a unitary gate U=e^{iAt} which is unitary. (t is another hyperparameter). Problem is how do we compute U? This is exactly the problem solved by Hamiltonian simulation, which is a classical problem in quantum computing, that is to approximate the evolution of a Hamiltonian system thorugh quantum gates to a desired precision. There exist several algorithms to tackle the problem, I won't go into the details. In my experiments I used an approach based on decomposing A as sum and simulate the decomposed hamiltonias but as said this is beyond the scope of my seminar.

Coming back to Quantum Phase Estimation, we encoded A in this way because we have way to approximate its action and because we preserve its eigenvalues. Indeed, the eigenvalues of A are related to eigenvalues of U by the equation written here, that is the eigenvales of U are the e^{i\lambda t }

Now, Qunatum Phase Estimation itself can be split into 3 stpes: superposition, controlled rotations and Inverse Quantum Fourier Trasnform.

Firstly we need to superimpose the qubits in the c-registers. This is necessary to exploit the phase kickback phenomenon.

Then, controlled rotations are applied. To see what happens let's assume for the moment that b is an eigenvecoto of U with eigenvalue e^{i2\pi \phi}. After that we will see that we will be able to recover the observation we'll make by expressing b in the eigenvector basis. So for the moment b is an eigenvector of U.

Now let's apply controlled rotations. If the c-register is 0, ket b is not affected, otherwise U is applied. What happens is highlighted in this equation. Each qubit in the c-register control the related power of U gate. In the second line we apply the fact that ket b is an eigenvector of U and phase kickback happens, namely we can factor out ket b and act as if the effect of the controlled gate only applies to the controlling gate as a phase gate. In the third line we rewrite in the c-register in the computational basis. What happened is that we transfered all the phase information related to the U gate to the c-register. However, this information is stored in the fourier basis

Indeed, Quantum Fourier Transform, as the standard Fourier transform maps a state in the computational basis to a state in the fourier basis. In the first equation I reported the result of applying QFT to a basis state.

What's of interste for us is that if we look at the circuit state after the controlled rotations we can see that the phase information is expressed in the c-registers in the fourier basis. So that, if we apply the inverse (which we can do as QFT is unitary) we can map the phase information in the computational basis. This by applying IQFT to the c-registers we get ket N\phi

Summing up, we assumed b to be an eigenvector of U, but we also encoded U as e^{iAt} so that we can relate phase and eigenvalues by the equation phi equals \lambda t / 2pi leading to state Psi 4

More generally, if we write b in the eigenvector basis we obtain the genral form of Psi4

## Ancilla qubit rotation
The eigenvalues are now encoded in the c-registers using basis encoding. Now define lambda_j tilda for simplicity. We now want to rotate the ancilla qubit in the following way (introducing a parameter C which governs the probability of the rotation).

Such a rotation is implemented through Ry, which are gates that rotates the qubits around the y axis. As we need to rotate ket 0 we can represent as shown in the figure. The angle of the rotation depends on C and on the encoded lambda tilde.

The way ancilla qubit rotation is implemented is the following: we map each possible c-register except for 0 to 1 through these X-gates sandwiches and we relate each state to its related Ry gate rotation. When lambda_j tilde are encoded only the rotations related to the encoded lambda_j tilde will trigger.

After that the ancilla qubit is measured. If 0 we discard the circuit result, as rotation did not happen, if 1 we're left with the result in Psi 6 where we can appreciate the that the eigenvales are inverted and we're approximating to the desired solution.

## Hyperparamters selection 
Now, we have introduced all the hyperparamters of the algorithm. Let's see how to set them. Thay are 3: the number of qubits in the c-registers, the evolution time t and the paramter C.

A first consideration about C is that it controls the probability of rotating the ancilla that we want to maximize. However C/lambda_j cannot exceed 1 so we set him to the minimum eigenvalue of A. To set n_c and t we have to condisfer that these parameters determines a discrete grid onto which the estimated eigenvalues of A are mapped to the nearest value on the grid. Tnhis grid has a scale determined by the fact that phase is ciclyc and so it has to be lower than 1, recalling the phase equation phi = lambda t/( 2pi) we can set t to be lower than 2pi /max eigval. The sensibility is governed by both N_c and it is finer as more qubits are used

## Uncomputation
Let's come to the uncomputation stage. We could think to be satisfied as we are because the state psi6 show the solution in the desired form in the b-registers. Problem is that we cannot access it becaues b-register are entangled with the c-register. What we need to do is to disentagle them by reversing the previous computations.

So firstly we apply the quantum fourier transofrm to c-registers to map to the fourier basis obtaining psi7. The we reversed the controlled rotations by applying controlled rotations with respect to U^{-1}. What we see is that in this way we're able to disentagle the qubits arriving to Psi8. Here it suffices to apply Hadamard gates to the c-registers and observe that the solution is already stored in the b-registers amplitudes.