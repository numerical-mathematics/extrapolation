C * * * * * * * * * * * * * * * * * * * * * * * * *
C --- DRIVER FOR DOPRI5 ON N-body problem
C * * * * * * * * * * * * * * * * * * * * * * * * *
cfeh dr_dop853 dop853
        include 'dop853.f'
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        PARAMETER (NDGL=2400,NRD=0)
        PARAMETER (LWORK=11*NDGL+8*NRD+21,LIWORK=NRD+21)
        DIMENSION Y(NDGL),WORK(LWORK),IWORK(LIWORK)
        EXTERNAL FNBOD,SOLOUT
C --- DIMENSION OF THE SYSTEM
        N=2400
        RPAR=1.0D-3
C --- OUTPUT ROUTINE (AND DENSE OUTPUT) IS USED DURING INTEGRATION
        IOUT=0
C --- INITIAL VALUES
        X=0.0D0
        pi=3.141592653589d0
        mbdy = 400
        rf1 = 4*pi/8
        rf2 = 2*pi/mbdy
        do i=1,mbdy
C          mass(i) = (0.3D0 +0.1D0*(cos(i*rf1)+1D0))/mbdy
          rad = 1.7D0+dcos(i*0.75D0)
          v = 0.22d0*dsqrt(rad)
          ci = dcos(i*rf2)
          si = dsin(i*rf2)
          ip = 6*(i-1)
          Y(ip+1) = rad*ci ! initial position
          Y(ip+2) = rad*si
          Y(ip+3) = 0.4D0*si
          Y(ip+4) = -v*si ! velocity is tangential
          Y(ip+5) = v*ci
          Y(ip+6) = 0.D0
        end do

C --- ENDPOINT OF INTEGRATION
        XEND=8.d-2
C --- REQUIRED (RELATIVE) TOLERANCE
        TOL=relative_tolerance
        ITOL=0
        RTOL=TOL
        ATOL=TOL
C --- DEFAULT VALUES FOR PARAMETERS
        DO 10 I=1,10
        IWORK(I)=0
  10    WORK(I)=0.D0   
        IWORK(5)=0
        IWORK(4)=1000
C --- CALL OF THE SUBROUTINE DOPRI8   
        CALL DOP853(N,FNBOD,X,Y,XEND,
     &                  RTOL,ATOL,ITOL,
     &                  SOLOUT,IOUT,
     &                  WORK,LWORK,IWORK,LIWORK,RPAR,IPAR,IDID)
C --- PRINT relative error
        OPEN(UNIT=12, FILE="reference.txt", ACTION="read")
        FERR = 0.d0
        do i=1,N
            READ(12,*) YREF
            FERR = FERR + ((Y(i)-YREF)/YREF)**2
        end do
        FERR = dsqrt(FERR/N)
        WRITE (6,97) FERR
 97     FORMAT(1X,'Error = ',E24.16)

C --- PRINT FINAL SOLUTION
        WRITE (6,99) X,Y(1),Y(2)
 99     FORMAT(1X,'X =',E24.16,'    Y =',2E24.16)
C --- PRINT STATISTICS
        WRITE (6,90) TOL
 90     FORMAT('       tol=',D8.2)
        WRITE (6,91) (IWORK(J),J=17,20)
 91     FORMAT(' fcn=',I5,' step=',I4,' accpt=',I4,' rejct=',I3)

C        OPEN(UNIT=12, FILE="solution.txt", ACTION="write",
C     &       STATUS="replace")
C        do i=1,N
C            WRITE (12,98) Y(i)
C        end do
C 98     FORMAT(1X,E24.16)

        STOP
        END
C
        SUBROUTINE SOLOUT (NR,XOLD,X,Y,N,CON,ICOMP,ND,
     &                                          RPAR,IPAR,IRTRN,XOUT)
C --- PRINTS SOLUTION AT EQUIDISTANT OUTPUT-POINTS
C --- BY USING "CONTD8", THE CONTINUOUS COLLOCATION SOLUTION
        IMPLICIT REAL*8 (A-H,O-Z)
        DIMENSION Y(N),CON(8*ND),ICOMP(ND)
        IF (NR.EQ.1) THEN
           WRITE (6,99) X,Y(1),Y(2),NR-1
           XOUT=0.1D0
        ELSE
 10        CONTINUE
           IF (X.GE.XOUT) THEN
              WRITE (6,99) XOUT,CONTD8(1,XOUT,CON,ICOMP,ND),
     &                     CONTD8(2,XOUT,CON,ICOMP,ND),NR-1
              XOUT=XOUT+0.1D0
              GOTO 10
           END IF
        END IF
 99     FORMAT(1X,'X =',F5.2,'    Y =',2E18.10,'    NSTEP =',I4)
        RETURN
        END
C
        SUBROUTINE FNBOD(N,X,Y,F,RPAR,IPAR)
C --- RIGHT-HAND SIDE OF N-body problem
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        DIMENSION Y(N),F(N)
        mbdy = 400
        eps  = 1.d-4
        dmass =  1.d0
        do i=1,mbdy
           ip = 6*(i-1)
           F(ip+1) = Y(ip+4)
           F(ip+2) = Y(ip+5)
           F(ip+3) = Y(ip+6)
           f1 = 0D0; f2 = 0D0; f3 = 0D0
          do j=1,mbdy ! compute gravity force body j <- i
            if (j.ne.i) then
              jp = 6*(j-1)
              dist=eps+(Y(ip+1)-Y(jp+1))**2+(Y(ip+2)-Y(jp+2))**2+
     &              (Y(ip+3)-Y(jp+3))**2
              dist = dmass/(dist*sqrt(dist))
              f1 = f1+(Y(jp+1)-Y(ip+1))*dist
              f2 = f2+(Y(jp+2)-Y(ip+2))*dist
              f3 = f3+(Y(jp+3)-Y(ip+3))*dist
            end if
          end do
          F(ip+4) = f1
          F(ip+5) = f2
          F(ip+6) = f3
        end do
        RETURN
        END

