      	subroutine centered_diff_fort(u,D,dx,N,Nsq)
  		integer :: N
  		integer, intent(in) :: Nsq
	  	double precision, intent(in) :: u(Nsq), dx
	  	double precision, intent(out) :: D(Nsq)
	  	integer :: i,j
	  	do i = 2,N-1 
	      		do j = 2,N-1 
	          		D((i-1)*N+j) = (u((i-1)*N+j+1) + u((i-2)*N+j)
     &			+ u((i-1)*N+j-1) + u(i*N+j) - 4*u((i-1)*N+j))/dx**2
	      		end do 
	  	end do
C --- i=1,i=N
		do j=2,N-1
			D(j) = (u(j+1) + u((N-2)*N+j) + u(j-1) + u(N+j) - 4*u(j))/dx**2
			D((N-1)*N+j) = D(j)
		end do
C --- j=1,j=N
		do i=2,N-1
			D((i-1)*N+1) = (u((i-2)*N+1) + u((i-1)*N+2) + u((i-1)*N+N-1)
     &			 + u(i*N+1) - 4*u((i-1)*N+1))/dx**2
			D((i-1)*N+N) = D((i-1)*N+1)		
		end do
C --- edges
		D(1) = (u(2) + u(N+1) + u(N-1) + u((N-2)*N+1) - 4*u(1))/dx**2
		D(N) = D(1)
		D((N-1)*N+1) = D(1)
		D(N*N) = D(1)
      	end subroutine centered_diff_fort

        SUBROUTINE fnbruss(Y,t,F,N,Nsq)
C --- RIGHT-HAND SIDE OF N-body problem
		integer, intent(in) :: Nsq, N
		double precision, intent(in) :: t, Y(2*Nsq)
		double precision, intent(out) :: F(2*Nsq)
		double precision :: Du(Nsq), Dv(Nsq), U(Nsq), V(Nsq)
		double precision :: uval, vval, finhom, dx
		alpha=0.1
		dx = 1.D0/(N-1)
		U = Y(1:Nsq)
		V = Y((Nsq+1):2*Nsq)
		CALL centered_diff_fort(U,Du,dx,N, Nsq)
		CALL centered_diff_fort(V,Dv,dx,N, Nsq)			
		do i=1,Nsq
			uval = U(i)
			vval = V(i)			
			finhom = BRUSS2DInhom(i,t,N,dx)
			F(i) = 1+(uval**2)*vval-4.4*uval+alpha*Du(i)+finhom
			F(Nsq+i) = 3.4*uval-(uval**2)*vval+alpha*Dv(i)
        	end do
        end subroutine fnbruss

	function BRUSS2DInhom(i,t,N,dx)
		integer, intent(in) :: i, N
		integer :: row
		double precision, intent(in) :: t, dx
		BRUSS2DInhom = 0.D0
		row = (i-1)/N
		y = dx*row
		x = dx*(i-1-row*N)
   		if((x-0.3)**2+(y-0.6)**2<=0.01 .AND. t>=1.1) then
	   		BRUSS2DInhom = 5.D0
		end if

	end function BRUSS2DInhom

