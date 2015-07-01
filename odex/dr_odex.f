C * * * * * * * * * * * * * * * * * * * * * * * * *
C --- DRIVER FOR ODEX ON N-body problem
C * * * * * * * * * * * * * * * * * * * * * * * * *
compile odex
cfeh dr_odex odex
        include 'odex_load_balanced.f'
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        PARAMETER (NDGL=2400,KM=9,NRDENS=0,
     &     LWORK=NDGL*(KM+5)+5*KM+20+(2*KM*(KM+2)+5)*NRDENS,
     &     LIWORK=2*KM+21+NRDENS)
        DIMENSION Y(NDGL),WORK(LWORK),IWORK(LIWORK)
        EXTERNAL FNBOD,SOLOUT
C --- DIMENSION OF THE SYSTEM
        N=2400
        RPAR=1.D-1
C --- OUTPUT ROUTINE AND DENSE OUTPUT IS USED DURING INTEGRATION
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
        H=0.01D0   
C --- IF DENSE OUTPUT IS REQUIRED
        IWORK(8)=NRDENS
C --- CALL OF THE SUBROUTINE ODEX
        CALL ODEX(N,FNBOD,X,Y,XEND,H,
     &                  RTOL,ATOL,ITOL,
     &                  SOLOUT,IOUT,
     &                  WORK,LWORK,IWORK,LIWORK,RPAR,IPAR,IDID)
C --- PRINT relative error
        OPEN(UNIT=12, FILE="reference.txt", ACTION="read")
        FERR = 0.d0
        do i=1,N
            READ(12,*) YREF
            FERR = FERR + ((Y(i)-YREF)/YREF)**2
C            write(*,*) Y(i), YREF, ((Y(i)-YREF)/YREF)
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
        STOP
        END
C
C
        SUBROUTINE SOLOUT (NR,XOLD,X,Y,N,CON,NCON,ICOMP,ND,
     &                     RPAR,IPAR,IRTRN)
C --- PRINTS SOLUTION AT EQUIDISTANT OUTPUT-POINTS
C --- BY USING "CONTD5", THE CONTINUOUS COLLOCATION SOLUTION
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        DIMENSION Y(N),CON(NCON),ICOMP(ND)
        COMMON /INTERN/XOUT  
        IF (NR.EQ.1) THEN
           WRITE (6,99) X,Y(1),Y(2),NR-1
           XOUT=X+0.1D0
        ELSE
 10        CONTINUE
           IF (X.GE.XOUT) THEN
              SOL1=CONTEX(1,XOUT,CON,NCON,ICOMP,ND)
              SOL2=CONTEX(2,XOUT,CON,NCON,ICOMP,ND)
              WRITE (6,99) XOUT,SOL1,SOL2,NR-1
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
        eps  = 1.D-4
        dmass =  1.D0
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
