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

  L   = 256
  a   = 1.0
  BZ  = 5
  G   = 4

  D = BZ*L+1
  ALLOCATE(E(G))
  ALLOCATE(V_AUX(G,G))
  ALLOCATE(V(G,G))
  ALLOCATE(K(G,G))
  ALLOCATE(H(G,G))

  !# FOR EACH QUASIMOMENTUM (m)
  K = 0.0
  V = 0.0
  
  !# KINETIC ENERGY 
  DO m=-L/2,L/2
     i_ = 1
     !  m  = -L/2.0
     DO n=-(G-1)/2,(G-1)/2
        
        momentum   = (1.0*m/L - n)
        K(i_,i_)   = 0.5*momentum**2
        IF(i_+1.LE.G) THEN 
           V(i_,i_+1) = 0.125
           V(i_+1,i_) = 0.125
        END IF
        !        write(*,*) i_, (-1.0*BZ*L/2.0 + 1.0*BZ*(m-1.0)*L/D)/L,& 
        !             & real(K(i_,i_))
        i_=i_+1     
     END DO
     
     H = K + V
     !CALL WRITE_MATRIX(REAL(K))  
     CALL LAPACK_FULLEIGENVALUES(H,G,E,INFO)
     write(*,*) 1.0*m/L,real(K(1,1)),real(K(2,2)),real(K(3,3)),real(K(4,4)),real(K(5,5)),E
  END DO
  !CALL LAPACK_FULLEIGENVALUES(H,G*D,E,INFO)
  !DO m=1,size(E,1)
  !   WRITE(*,*) E(m)
  !END DO
END PROGRAM OL_SPECTRUM
