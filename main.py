import numpy as np
import matplotlib.pyplot as plt
from root_finder import RetardedTime
from time_plotter import TimePlotter,To_Plot

#Parameters for the simulation

#length of the oscilation
l = 1.0
#Angular frequency of the oscilation
w = 0.1
#The speed of light
c = 1.0
#A sample position at which to compute the retarded time for testing purposes
test_position = np.asarray([0.0,10.0,0.0])
#The current time at the field point
t = 3.0
#The z coordinate to sample.
z = 0.0

# The function describing the posisition of the charge over time.
def path(t):
	x = l*np.cos(w*t)
	y = l*np.sin(w*t)
	z = 0
	# x = 0
	# y = 0
	# z = l*np.cos(w*t)
	#Returns the vector of the position of the particle at time t
	return np.asarray([x,y,z])

def velocity(t):
	x = -w*l*np.sin(w*t)
	y = w*l*np.cos(w*t)
	z = 0
	# x = 0
	# y = 0
	# z = -w*l*np.sin(w*t)
	#Returns the vector of the velocity of the particle at time t
	return np.asarray([x,y,z])

def acceleration(t):
	x = -w**2*l*np.cos(w*t)
	y = -w**2*l*np.sin(w*t)
	z = 0
	# x = 0
	# y = 0
	# z = -w**2*l*np.cos(w*t)
	#Returns the vector of the velocity of the particle at time t
	return np.asarray([x,y,z])

def main():
	global w
	#Creates a new retarded time class
	rt = RetardedTime(path,test_position, t, c,velocity=velocity)
	#Makes a new plotter to plot the retarded time
	tp = TimePlotter(rt, -100,100,100,-100,100,100,t,z)
	#Makes a color map of the retarded time
	#tp.make_image(To_Plot.ELECTRIC_FIELD)
	#Generates a time animation of the retarded time
	#tp.generate_time_animation(0,4,50,To_Plot.ELECTRIC_FIELD)
	angles = [90,75,45,15,0]
	r=950
	fig, axs = plt.subplots(len(angles),1,sharex=True)
	phi = np.deg2rad(90)
	for i,j in zip(axs,angles):
		i.set_title(rf'Radial Poynting Vector Over Time at ${90-j}^\circ$')
		i.set_ylabel(r'$S_r$ (normalized)')
		w=0.1
		tp.time_line_plot(np.deg2rad(j),r*(2*c*np.pi/w),tf=2*np.pi/w, ax = i,label=rf'$\frac{{\omega l}}{{c}} ={w} $',phi=phi)
		w = 0.4
		tp.time_line_plot(np.deg2rad(j),r*(2*c*np.pi/w),tf=2*np.pi/w, ax = i,label=rf'$\frac{{\omega l}}{{c}} ={w} $',phi=phi)
		w = 0.8
		tp.time_line_plot(np.deg2rad(j),r*(2*c*np.pi/w),tf=2*np.pi/w, ax = i,label=rf'$\frac{{\omega l}}{{c}} ={w} $',phi=phi)
	axs[-1].legend()
	axs[-1].set_xlabel(r'Phase $\omega t$')
	plt.show()
	ws = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.999]
	points = [100,100,100,100,100,100,100,100,100,100]
	ax = plt.subplot()
	radius = 950
	for i,j in zip(ws,points):
		w = i
		tf = 2*np.pi/w
		tp.angular_dependence_plot(radius,numpoints=j,tf=tf,ax=ax,label=rf'$\frac{{\omega l}}{{c}} ={w} $',phi=phi)
	ax.legend(loc='lower right')
	ax.set_xlabel(r'Angle $\theta$')
	ax.set_ylabel(r'$\frac{dP}{d\Omega}$ (normalized)')
	ax.set_title(r'Angular Power Distribution as a Function of $\frac{\omega l}{c}$')
	plt.savefig("power_dist.pdf")
	plt.show()

#Calls the main function which is the entry point to this program.
main()
