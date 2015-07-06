Variational Approach for conformation dynamics (VAC)
====================================================

This package contains basis sets, estimators and solvers for the variational approach for 
conformation dynamics, a theory that has been proposed in [1] and was further developed in
[2] and [3]. The variational approach is analogous to the Ritz method [4] that is 
employed in computational quantum chemistry. It differs in the way how the involved
matrices are computed and in the meaning of the involved operators, eigenfunctions and 
eigenvalues - see [3] for a comparison. 

Roughly, the idea of the VAC is as follows: Given a (classical) 
molecular dynamics trajectory with configurations {x_1, ..., x_T}, and a 
set of basis functions defined on the space of configurations {chi_1(x), ..., chi_n(x)},
we compute the two correlation matrices:

c_ij (0)   = < chi_i(x_t) chi_j(x_t) >_t
c_ij (tau) = < chi_i(x_t) chi_j(x_t+tau) >_t

where < . >_t is average over time t. Of course this can be generalized to many trajectories.
Then we solve the generalized eigenvalue problem

C(tau) r = C(0) r l(tau)

where the eigenvalues l(tau) approximate the dominant eigenvalues of the Markov propagator
or Markov backward propagator of the underlying dynamics. The corresponding eigenfunction
of the backward propagator is approximated by

psi(x) = sum_i r_i chi_i(x)

This package aims at providing code to help addressing a number of key problems:

1. Basis sets for molecular dynamics (MD), and in particular protein dynamics. See [5]_ for a
   first approach in this direction.

2. Estimators for the corration matrices C(0), C(tau). The trivial time-average that is usually
   employed has a number of problems especially for many short simulation trajectories that are
   initiated far from the equilibrium distribution (the usual case!).

3. Solvers for accurately solving the eigenvalue problem above, even for huge basis sets. 

At this time only a few of these functionalities are implemented and we will go step by step.
This package will undergo heavy development and there is currently no date for an official
release, so don't be surprised if the API (the look + feel of functions and classes) change.
At the moment this package is purely intended for development purposes, so use it at your own
risk.

References
----------
[1] Noe, F. and Nueske, F. (2013): A variational approach to modeling slow processes in stochastic dynamical systems. SIAM Multiscale Model. Simul. 11, 635-655.

[2] Nueske, F., Keller, B., Perez-Hernandez, G., Mey, A.S.J.S. and Noe, F. (2014) Variational Approach to Molecular Kinetics. J. Chem. Theory Comput. 10, 1739-1752.

[3] Perez-Hernandez, G., Paul, F., Giorgino, T., De Fabritiis, G. and Noe, F. (2013) Identification of slow molecular order parameters for Markov model construction. J. Chem. Phys. 139, 015102.

[4] Ritz, W. (1909): Ueber eine neue Methode zur Loesung gewisser Variationsprobleme der mathematischen Physik. J. Reine Angew. Math., 135, 1–61.

[5] Vitalini, F., Noé, F. and Keller, B. (2015): A basis set for peptides for the variational approach to conformational kinetics. (In review).

