!!$PROGRAM AVGPOTENTIAL
!!$
!!$  USE physical_constants
!!$  IMPLICIT NONE
!!$
!!$  DOUBLE PRECISION :: MASS, a_s,N_ATOMS
!!$
!!$  DOUBLE PRECISION :: V_0,d_0,PHI_0
!!$  DOUBLE PRECISION :: DELTA_V,DELTA_d,OMEGA,PHI_T
!!$  
!!$  DOUBLE PRECISION :: l,k,l_HO,E_HO,OMEGA_X
!!$  
!!$  DOUBLE PRECISION :: N_GPE,ENERGY_SCALING
!!$
!!$  DOUBLE PRECISION :: TIME,d,DL,T,E_R
!!$  DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: X,V_AVG
!!$
!!$  INTEGER N,I,N_X,N_T,INFO
!!$
!!$  
!!$  N_T   = 512
!!$
!!$  MASS  = 87*AMU
!!$
!!$  ! OPTICAL LATTICE PARAMETERS
!!$  d_0   = 2.05     ! SPACING IN MICRONS
!!$  d     = d_0*25   ! SPATIAL DOMAIN [-L:L]
!!$  E_R   = hbar*hbar*PI*PI/(2*MASS*d_0*d_0*1E-12) ! this is the unit of energy FOR BLOCH spectrum
!!$  V_0   = 1.0      ! OPTICAL LATTICE DEPTH IN UNITS OF E_R
!!$  PHI_0 = 0.0      ! OPTICAL LATTICE SPATIAL PHASE
!!$
!!$
!!$  ! EXTERNAL TRAPPING
!!$  OMEGA_X = 2*PI*50.0
!!$  l       = 20.0
!!$  k       = 20.0
!!$  E_HO    = HBAR*OMEGA_X ! THIS IS THE ENERGY UNIT FOR THE GPE
!!$
!!$
!!$  !DRIVING PARAMETERS
!!$  OMEGA   = 1.0 
!!$  PHI_T   = 0.0 
!!$  DELTA_d = 0.0 ! in microns
!!$  
!!$  
!!$  !NON-LINEAR  COUPLING
!!$  N_GPE = 2*a_s*N_atoms*l_HO*sqrt(2.0*l*k)
!!$
!!$  N_X = 2048
!!$  ALLOCATE(V_AVG(N_X))
!!$  ALLOCATE(X(N_X))
!!$  CALL OPTICAL_LATTICE_POTENTIAL(N_X,MASS,V_0,d_0,d,PHI_0,OMEGA,PHI_T,DELTA_d, OMEGA_X, V_AVG,ENERGY_SCALING,INFO)  
!!$
!!$  DO I = 1,N_X
!!$     X(I) = -d  + I*2.0*D/N_X
!!$     WRITE(*,*) X(I),V_AVG(I) 
!!$  END DO
!!$
!!$END PROGRAM

SUBROUTINE OPTICAL_LATTICE_POTENTIAL(N, V_AVG,INFO)  

 ! write(*,*) '# E_R :',  E_R, E_R/(hbar*2*pi)
 ! write(*,*) '# E_HO: ', hbar*2*pi*5.0,5.0
 ! write(*,*) '# l_HO: ', sqrt(87*amu*2*pi*5/hbar)
 ! write(*,*) '# N=2 a N sqrt(m  omega_x /hbar) sqrt(2 l k): ', 2.0*30*a_0*5E4*sqrt(87*amu*2*pi*5/hbar)*sqrt(2.0)*20
  USE physical_constants
  USE optical_lattice_parameters
  IMPLICIT NONE
  INTEGER,                            INTENT(IN)    :: N
  !DOUBLE PRECISION,                 INTENT(IN)    :: MASS,V_0,d_0,d,PHI_0,OMEGA,PHI_T,DELTA_d,OMEGA_X
  !DOUBLE PRECISION,                   INTENT(OUT)   :: ENERGY_SCALING
  DOUBLE PRECISION, DIMENSION(N+1), INTENT(OUT)   :: V_AVG
  INTEGER,                            INTENT(INOUT) :: INFO

  INTEGER I
  DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE:: X,V_TRAP
  DOUBLE PRECISION :: DL,T,ENERGY_SCALING!,E_HO,E_R,T

  ALLOCATE(X(N+1))
  ALLOCATE(V_TRAP(N+1))


  DO I=1,N+1
     X(I) = -d  + I*2.0*D/N
  END DO

  V_AVG = 0.0
  DO I = 1,N_T
     T     = I*(2*PI/OMEGA)/N_T
     DL    = DELTA_d*SIN(OMEGA*T+PHI_T)     
     V_AVG = V_AVG - 20.0*COS(2*PI*X/(d_0+DL)+PHI_0)/N_T
  END DO

  V_TRAP = 0.5*MASS*OMEGA_X*OMEGA_X*X*X*1E-12
  
  V_AVG = 1.0*V_AVG + V_TRAP/E_R

  ENERGY_SCALING = E_R/E_HO

  V_AVG = V_AVG*ENERGY_SCALING
 
  


END SUBROUTINE OPTICAL_LATTICE_POTENTIAL
