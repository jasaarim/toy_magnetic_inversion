C File integrand.f
	subroutine response_point(F_hat, F_abs, x, y, response)
Cf2py intent(in) :: F_hat(2),F_abs,x,y
Cf2py intent(out) :: response
	
	real*8 F_hat(2), F_abs, x, y, response
	real*8 PI,r_abs, r_hat(2), factor

	PI = 3.14159265358979323

	r_abs = sqrt(x**2 + y**2)
	r_hat(1) = x / r_abs
	r_hat(2) = y / r_abs
	factor = 2*(F_hat(1)*r_hat(1)+F_hat(2)*r_hat(2))**2 -1
	response = F_abs * factor / ( 2 * PI * r_abs**2 ) 

	end
