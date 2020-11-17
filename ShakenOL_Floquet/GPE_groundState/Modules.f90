MODULE physical_constants
  IMPLICIT NONE
  DOUBLE PRECISION, PARAMETER :: pi           = 4.0*ATAN(1.0)
  DOUBLE PRECISION, PARAMETER :: e            = 1.602176462E-19
  DOUBLE PRECISION, PARAMETER :: h_P          = 6.62606957E-34
  DOUBLE PRECISION, PARAMETER :: hbar         = 1.054571800E-34
  DOUBLE PRECISION, PARAMETER :: mu_B         = 9.274009994E-24
  DOUBLE PRECISION, PARAMETER :: k_B          = 1.3806488E-23
  DOUBLE PRECISION, PARAMETER :: mu_cero      = 12.566370614E-7
  DOUBLE PRECISION, PARAMETER :: epsilon_cero = 8.854187817E-12 
  DOUBLE PRECISION, PARAMETER :: amu          = 1.660538921E-27
  DOUBLE PRECISION, PARAMETER :: g_t          = 9.8
  DOUBLE PRECISION, PARAMETER :: SB_ct        = 5.6704E-8
  COMPLEX*16,       PARAMETER :: J_IMAG       = DCMPLX(0.0,1.0)
  DOUBLE PRECISION, PARAMETER :: speedoflight = 299792458.0
  DOUBLE PRECISION, PARAMETER :: a_0          = 5.29E-11
  DOUBLE PRECISION            :: TOTAL_TIME
END MODULE physical_constants


MODULE optical_lattice_parameters
  
  ! BEC PARAMETERS
  USE physical_constants
  IMPLICIT NONE
  DOUBLE PRECISION, PARAMETER :: N_ATOMS = 5E4
  INTEGER                     :: N_X 
  INTEGER, PARAMETER          :: N_T = 512
  DOUBLE PRECISION, PARAMETER :: MASS    = 87*AMU
  DOUBLE PRECISION, PARAMETER :: a_s     = 30*a_0
  
  !     OPTICAL LATTICE PARAMETERS
  DOUBLE PRECISION, PARAMETER ::   d_0   = 2.05              ! SPACING IN MICRONS
  DOUBLE PRECISION, PARAMETER ::   d     = d_0*50            ! SPATIAL DOMAIN [-L:L]
  DOUBLE PRECISION, PARAMETER ::   E_R   = hbar*hbar*PI*PI/(2*MASS*d_0*d_0*1E-12) ! this is the unit of energy FOR BLOCH spectrum
  DOUBLE PRECISION, PARAMETER ::   V_0   = 0.0               ! OPTICAL LATTICE DEPTH IN UNITS OF E_R
  DOUBLE PRECISION, PARAMETER ::   PHI_0 = 0.0               ! OPTICAL LATTICE SPATIAL PHASE
  
   
  ! EXTERNAL TRAPPING
  DOUBLE PRECISION, PARAMETER ::   OMEGA_X = 2*PI*5.0
  DOUBLE PRECISION, PARAMETER ::   l_       = 20.0
  DOUBLE PRECISION, PARAMETER ::   k_       = 20.0
  DOUBLE PRECISION, PARAMETER ::   E_HO    = HBAR*OMEGA_X    ! THIS IS THE ENERGY UNIT FOR THE GPE
  DOUBLE PRECISION, PARAMETER ::   l_HO    = sqrt(HBAR/(MASS*OMEGA_X))
  
  
  !     DRIVING PARAMETERS
  DOUBLE PRECISION, PARAMETER ::   OMEGA   = 1.0 
  DOUBLE PRECISION, PARAMETER ::   PHI_T   = 0.0 
  DOUBLE PRECISION, PARAMETER ::   DELTA_d = 0.15             ! in microns
  
  
  !NON-LINEAR  COUPLING
  DOUBLE PRECISION, PARAMETER ::   N_GPE = 2*a_s*N_atoms*l_HO*sqrt(2.0*l_*k_)
  

!  DOUBLE PRECISION :: ENERGY_SCALING
  
!  DOUBLE PRECISION ::  DX! DX = (d*1E-6/l_HO)/N
!  DOUBLE PRECISION ::  DT! DT = (1/OMEGA_X)/256.0
  
END MODULE optical_lattice_parameters
