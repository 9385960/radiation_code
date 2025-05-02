import numpy as np
from scipy.optimize import newton

#A new class for computing the retarded time at a point
class RetardedTime:
	#When making a new Retarded time class, it takes in a path for the particle trajectory, a field point for the location to determine the retarded time at, the local time at the field point, and the speed of light
	def __init__(self,path,field_point = np.asarray([0.0,0.0,0.0]),t=0.0,c=3e8,q=1.0, velocity = None, acceleration = None):
		#Sets the class variables to be used later in the code
		#The path is a function handle to the path the particle takes
		self.p = path
		#C is simply the speed of light
		self.c = c
		#The time at the field point
		self.t = t
		#The position of the field point
		self.f = field_point
		#The charge of the particle in motion
		self.q = q
		#The velocity of the particle as a function of time
		self.v = velocity
		#The acceleration fo the particle as a function of time
		self.a = acceleration
	
	#This is the definition of the retarded time that needs to be solved. We must find the tr that makes this equation 0.
	def retarded_time_def(self,tr):
		return self.t-tr-np.linalg.norm(self.f-self.p(tr))/self.c
	
	#Function to compute the retarded position
	def retarded_position(self,tr):
		return self.p(tr)
	
	#Function to compute the lienhard-weichert potentials
	def lw_potential(self):
		#Compute the retarded time
		rt = self.compute_retarded_time()
		#Compute the vector R defined in melia
		r = self.f-self.retarded_position(rt)
		#gets the magnitude of r
		r_mag = np.linalg.norm(r)
		#Get the n_hat vector defined in melia
		n_hat = r/r_mag
		#Computes the velocity of the particle and divides by c
		if self.v is None:
			v = (self.p(rt+1e-5)-self.p(rt-1e-5))/(2e-5)
			beta = v/self.c
		else:
			beta = self.v(rt)/self.c
		#Computes the acceleration of the particle and divides by c
		if self.a is None:
			a = (self.p(rt+1e-5)-2*self.p(rt)-self.p(rt-1e-5))/((1e-5)**2)
			beta_dot = a/self.c
		else:
			beta_dot = self.a(rt)/self.c
		#Calculates the electric and magnetic vector potentials using the previously calculated values
		electric_potential = self.q/((1-n_hat @ beta)*r_mag)
		magnetic_potential = electric_potential*beta
		#Calculates the electric and magnetic fields
		e_field = self._e_field(n_hat,beta,r_mag,beta_dot)
		b_field= self._b_field(n_hat,e_field)
		#Calculates the poynting vector
		s = self.c/(4*np.pi)*np.cross(e_field,b_field)
		#Returns the computed values
		return (rt,electric_potential,magnetic_potential,e_field,b_field,s)
		
	
	#This function uses a newton raphson root finder to solve the above function.
	def compute_retarded_time(self):
		#Compute an initial time
		initial_time = np.linalg.norm(self.f-self.p(self.t))/self.c
		#Computes the retarded time starting at an initial point at a time before the current field point time.
		tr = newton(self.retarded_time_def,initial_time,tol=1e-10,maxiter=100)
		#A sanity check. If the correct tr is found, the following relation should hold.
		check = np.linalg.norm(self.f-self.p(tr))/(self.t-tr)
		if np.abs(self.c-check) > 1e-3:
			print("Error Speed of light incorrect: ",self.c - check)
		if tr > self.t:
			print("Found advenced time not retarded time.")
		#Returns the retarded time
		return tr
	
	#Sets the field point should the tr be desired at some other location than initially given
	def set_point(self, field_point):
		self.f = field_point
		
	#Sets the field point time should a new time be desired.
	def set_time(self,t):
		self.t = t

	#Computes the electric field at tr as described in melia
	def _e_field(self,n_hat,beta,r_mag,beta_dot):
		#Computes n_hat-beta since it is used multiple times in the calculation
		a = n_hat-beta
		#Similar to above, (1-n_hat.beta)^3 * R is used several times in the calculation
		b = ((1-n_hat@beta)**3)*r_mag
		#Computes the numerator of he first fraction in melia
		frac1_num = a*(1-beta@beta)
		#Computes the denominator of the first fraction in melia
		frac1_denom = b*r_mag
		#Computes the numerator of the second fraction in melia
		frac2_num = np.cross(a,beta_dot)
		frac2_num = np.cross(n_hat,frac2_num)
		#Computes the denominator of the second fraction in melia
		frac2_denom = self.c*b
		#Puts it all together to calculate the e_field
		e_field= self.q*((frac1_num/frac1_denom)+(frac2_num/frac2_denom))
		return e_field

	#Computes the B-field given in melia
	def _b_field(self,n_hat,e_field):
		return np.cross(n_hat,e_field)