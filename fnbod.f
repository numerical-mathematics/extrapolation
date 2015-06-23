        SUBROUTINE INIT_FNBOD(N,Y)
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        intent(in) :: N
        double precision, intent(out) :: Y(N)
C --- INITIAL VALUES
        pi=3.141592653589d0
        mbdy = N/6
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
        RETURN
        END 

        SUBROUTINE FNBOD(Y,X,F,N)
C --- RIGHT-HAND SIDE OF N-body problem
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        double precision, intent(in) :: X, Y(N)
        double precision, intent(out) :: F(N)
        mbdy = N/6
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