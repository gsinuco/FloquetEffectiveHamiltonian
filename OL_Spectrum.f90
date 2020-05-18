!#8TH OF MAY 2020
!# GERMAN SINUCO 
!# PROGRAM TO EVALUATE THE BLOCH BANDS OF A PERIODIC POTENTIAL


MODULE SUBINTERFACE_LAPACK

  IMPLICIT NONE
  PUBLIC 
  INTERFACE
     SUBROUTINE WRITE_MATRIX(A)
       DOUBLE PRECISION, DIMENSION(:,:) :: A
     END SUBROUTINE  WRITE_MATRIX

     SUBROUTINE WRITE_MATRIX_INT(A)
       INTEGER, DIMENSION(:,:) :: A
     END SUBROUTINE  WRITE_MATRIX_INT
     
  END INTERFACE
END MODULE SUBINTERFACE_LAPACK


PROGRAM OL_SPECTRUM

  USE SUBINTERFACE_LAPACK
  IMPLICIT NONE
  INTEGER L,m,n,info,BZ,D,i_,G,j_
  DOUBLE PRECISION :: a,k_,g_,momentum

  COMPLEX*16,       DIMENSION(:,:), ALLOCATABLE :: V,K,H,V_aux
  DOUBLE PRECISION, DIMENSION(:),   ALLOCATABLE :: e

  L   = 32
  a   = 1.0
  BZ  = 6
  G   = 5

  D = BZ*L+1
  ALLOCATE(E(G*D))
  ALLOCATE(V_AUX(D,D))
  ALLOCATE(V(G*D,G*D))
  ALLOCATE(K(G*D,G*D))
  ALLOCATE(H(G*D,G*D))

  !# FOR EACH QUASIMOMENTUM (m)
  K = 0.0
  V = 0.0
  
  !# KINETIC ENERGY 
  i_ = 1
  DO n=-(G-1)/2,(G-1)/2

     DO m=1,D
        
        momentum = ((-1.0*BZ*L/2.0 + 1.0*BZ*(m-1.0)*L/D)/L - n)
        K(i_,i_) = momentum**2
!        write(*,*) i_, (-1.0*BZ*L/2.0 + 1.0*BZ*(m-1.0)*L/D)/L,& 
!             & real(K(i_,i_))
        i_=i_+1
     END DO
!     write(*,*)
!     write(*,*)
  END DO
  !# POTENTIAL ENERGY 
  i_ = 1
  DO m=1,D     
     V_AUX(m,m) = 0.25
  END DO
  DO n=1,G-1
     i_ = 1 + (n-1)*D
     j_ = 1 + (n-1)*D + D
     !write(*,*) i_,":",i_+D-1,",", j_,":",j_+D-1
     V(i_:i_+D-1,j_:j_+D-1) = V_AUX   
  END DO
  V = V + TRANSPOSE(CONJG(V))
  

  H = K + V
  !CALL WRITE_MATRIX(REAL(H))
  
  CALL LAPACK_FULLEIGENVALUES(H,G*D,E,INFO)
  DO m=1,size(E,1)
     WRITE(*,*) E(m)
  END DO
END PROGRAM OL_SPECTRUM
