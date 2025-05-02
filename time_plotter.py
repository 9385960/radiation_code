import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from enum import Enum
from scipy.integrate import simpson,trapezoid

#Enum of types of plots(Picking a vecotor field will result in the magnitude of the vector to be plotted)
class To_Plot(Enum):
	RETARDED_TIME = 1
	ELECTRIC_POTENTIAL = 2
	MAGNETIC_POTENTIAL = 3
	ELECTRIC_FIELD = 4
	MAGNETIC_FIELD = 5
	POYNTING_VECTOR = 6

# A Class for generating plots of the retarded time.
class TimePlotter:
	# The time plotter requires a RetardedTime obect which will compute the retarded time for us, the bounds of the area to be plotted as well as the number of pixels in each direction, 
	#the time to compute generate the plot at, and the z coordinate to generate the plot at.
	def __init__(self, rt, xmin, xmax, xn, ymin, ymax, yn, t, z):
		#The x-values that the retarded time needs to be sampled at
		self.x_vals = np.linspace(xmin,xmax,xn)
		#The y-values that the retarded time needs to be sampled at
		self.y_vals = np.linspace(ymin,ymax,yn)
		#The time for all the field points
		self.t = t
		#The z-coordinate of all the field points
		self.z = z
		#The number of x points
		self.xn = xn
		#The number of y points
		self.yn = yn
		#The RetardedTime object which computes the retarded time for the plots
		self.rt = rt
		#Sets the time of the Retarded Time Objecet to the time of interest
		rt.set_time(t)

	def get_poynting_vector_field_snapshot(self,xmin,xmax,xnum,ymin,ymax,ynum,zmin,zmax,znum,t=0):
		xvals = np.linspace(xmin,xmax,xnum)
		yvals = np.linspace(ymin,ymax,ynum)
		zvals = np.linspace(zmin,zmax,znum)
		return


	def get_poynting_time_series_at_point(self,point,numpoints,tf):
		result = []
		self.rt.set_point(point)
		times = np.linspace(0,tf,numpoints)
		for i in range(numpoints):
			self.rt.set_time(times[i])
			em = self.rt.lw_potential()
			s = em[-1]
			result.append(s)
		return (times,np.asarray(result))

	def angular_dependence_plot(self,radius = 100, numpoints=1000, tf = 10,ax = None,label =None,fmt='-',phi = 0):
		result = np.zeros(numpoints)
		angles = np.linspace(0,np.pi,numpoints)
		for i in range(numpoints):
			position = np.asarray([radius*np.sin(angles[i])*np.cos(phi),radius*np.sin(angles[i])*np.sin(phi),radius*np.cos(angles[i])])
			r_hat = position/np.linalg.norm(position)
			(times,poynting) = self.get_poynting_time_series_at_point(position,numpoints,tf)
			x_time_avg= simpson(poynting[:,0],times)
			y_time_avg= simpson(poynting[:,1],times)
			z_time_avg= simpson(poynting[:,2],times)
			time_avg = np.asarray([x_time_avg,y_time_avg,z_time_avg])/(times[-1]-times[0])
			result[i] = time_avg@r_hat * radius*radius
		max = np.max(result)
		if max != 0:
			result = result/max
		if ax != None:
			ax.plot(angles,result,fmt,label=label)

	def time_line_plot(self, angle = 0, radius = 100, numpoints=1000, tf = 10, ax = None,label =None,fmt='-', phi = 0):
		result = np.zeros(numpoints)
		position = np.asarray([radius*np.sin(angle)*np.cos(phi),radius*np.sin(angle)*np.sin(phi),radius*np.cos(angle)])
		r_hat = position/np.linalg.norm(position)
		self.rt.set_point(position)
		times = np.linspace(0,tf,numpoints)
		for i in range(numpoints):
			self.rt.set_time(times[i])
			em = self.rt.lw_potential()
			s = em[-1]
			result[i] = np.dot(s,r_hat)
		max = np.max(result)
		if max != 0:
			result = result/max
		if ax != None:
			phase = np.linspace(0,2*np.pi,numpoints)
			ax.plot(phase,result,fmt,label=label)
	

	#Computes the map of the retarded time.
	def compute_values(self,to_plot = To_Plot.RETARDED_TIME):
		#Makes a new 2d-array which will contain the image
		result = np.zeros((self.xn,self.yn))
		#Itterates over the array.
		for i in range(self.xn):
			for j in range(self.yn):
				#For each point in the array, we set the field_point location
				field_point = np.asarray([self.x_vals[j],self.y_vals[i], self.z])
				self.rt.set_point(field_point)
				#Computes the potentials
				lw = self.rt.lw_potential()
				#Stores the desired result in the 2d-array
				if to_plot is To_Plot.RETARDED_TIME:
					result[i][j] = lw[0]
				elif to_plot is To_Plot.ELECTRIC_POTENTIAL:
					result[i][j] = lw[1]
				elif to_plot is To_Plot.MAGNETIC_POTENTIAL:
					result[i][j] = np.linalg.norm(lw[2])
				elif to_plot is To_Plot.ELECTRIC_FIELD:
					result[i][j] = np.linalg.norm(lw[3])
				elif to_plot is To_Plot.MAGNETIC_FIELD:
					result[i][j] = np.linalg.norm(lw[4])
				else:
					result[i][j] = lw[0]
		#Returns the 2d-array of the desired result.
		return result
	
	#Sets the time of the computation
	def set_time(self, t):
		self.t = t
		self.rt.t = t
	
	#Generates an image and saves it as colormap
	def make_image(self,to_plot = To_Plot.RETARDED_TIME, savefig = False, savename = "color_map.pdf"):
		#Computes the values
		image = self.compute_values(to_plot)
		#Plots the image
		plt.imshow(image, cmap='viridis',extent = [self.x_vals[0],self.x_vals[-1],self.y_vals[0],self.y_vals[-1]])
		#Displays the color bar
		plt.colorbar()
		#Makes the plot title
		plt.title(r'Plot of $\tilde{t}(\vec{x},t)$ over the xy-plane.')
		#Saves the image
		if savefig:
			plt.savefig(savename)
		#Returns the image
		return plt
	
	#The animation update step for generating gifs
	def _time_animation_update(self,frame,to_plot = To_Plot.RETARDED_TIME):
		#Updates the time of the simulation
		self.set_time(self.times[frame])
		#Generates a new plot at the new time
		data = self.compute_values(to_plot)
		#Sets the plot so that it has the 
		self.im_plot.set_array(data)
		#Re-maps the colorbar so that the correct bounds are shown
		vmin = np.min(data)
		vmax = np.max(data)
		self.im_plot.set_clim(vmin,vmax)
	
	#Generates a gif over time of the xy plane
	def generate_time_animation(self, tmin,tmax,tn,to_plot = To_Plot.RETARDED_TIME):
		#Makes a new plot
		self.fig, self.ax = plt.subplots()
		#Sets the current time of the simulation
		self.set_time(tmin)
		#Creates all the times at which to compute the retarded time for
		self.times,interval = np.linspace(tmin,tmax,tn,retstep = True)
		#Generates the first frame of the animation
		self.im_plot = self.ax.imshow(self.compute_values(), cmap="viridis",extent = [self.x_vals[0],self.x_vals[-1],self.y_vals[0],self.y_vals[-1]])
		self.fig.colorbar(self.im_plot)
		plt.subplots_adjust(bottom=0.3)
		#Adds a text description to the animation
		self.fig.text(0.01,0.1, "A gif of the retarded time: t' xy-plane from t = "+str(tmin)+" to t = "+str(tmax)+" with c = "+str(self.rt.c)+" and z = "+ str(self.z)+".",fontsize=8)
		#Generates the animation. This may take a considerable amount of time depending on the anmation parameters
		ani = animation.FuncAnimation(self.fig,self._time_animation_update,frames = tn, interval = interval*1000,fargs=(to_plot,))
		#Saves the animation
		ani.save("time_animation.gif", writer="pillow")