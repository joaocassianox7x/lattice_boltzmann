import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
import tqdm

matplotlib.use('TkAgg')

def main():
	""" Finite Volume simulation """

	carro = cv2.cvtColor(cv2.imread('carro.png'),cv2.COLOR_BGR2GRAY)
	zeros = np.zeros((227,500))
	carro = np.concatenate((zeros,carro,zeros),axis=1)
	carro = carro*(carro<100)
	carro = cv2.resize(carro,(0,0),fx=0.5,fy=0.5)
	cylinder = carro>0

	Nx          = np.shape(cylinder)[1]    # resolution x-dir
	Ny          = np.shape(cylinder)[0]    # resolution y-dir

	rho0                   = 100    # average density
	tau                    = 0.6    # collision timescale
	Nt                     = 10000   # number of timesteps
	plotRealTime = True # switch on for plotting as the simulation goes along

	# Lattice speeds / weights
	NL = 9
	idxs = np.arange(NL)
	cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
	cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
	weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1

	# Initial Conditions
	F = np.ones((Ny,Nx,NL)) #* rho0 / NL
	np.random.seed(42)
	#F += 0.1*np.random.randn(Ny,Nx,NL)
	X, Y = np.meshgrid(range(Nx), range(Ny))
	F[:227,:,3] += 1
	rho = np.sum(F,2)
	for i in idxs:
		F[:,:,i] *= rho0 / rho

	# Cylinder boundary
	X, Y = np.meshgrid(range(Nx), range(Ny))

	# Prep figure
	fig = plt.figure(figsize=(12,8), dpi=80)

	# Simulation Main Loop
	for it in tqdm.tqdm(range(Nt)):

		# Drift
		for i, cx, cy in zip(idxs, cxs, cys):
			F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
			F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)


		# Set reflective boundaries
		bndryF = F[cylinder,:]
		bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]


		# Calculate fluid variables
		rho = np.sum(F,2)
		ux  = np.sum(F*cxs,2) / rho
		uy  = np.sum(F*cys,2) / rho


		# Apply Collision
		Feq = np.zeros(F.shape)
		for i, cx, cy, w in zip(idxs, cxs, cys, weights):
			Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )

		F += -(1.0/tau) * (F - Feq)

		# Apply boundary
		F[cylinder,:] = bndryF


		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and (it % 2) == 0) or (it == Nt-1):
			plt.cla()
			ux[cylinder] = 0
			uy[cylinder] = 0
			vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1,axis=1))
			vorticity[cylinder] = np.nan
			cmap = plt.cm.bwr
			cmap.set_bad('black')
			plt.imshow(vorticity, cmap='bwr')
			plt.clim(-.1, .1)
			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.set_aspect('equal')
			plt.savefig(f"figures/{it:05d}.png")


	plt.savefig('latticeboltzmann.png',dpi=240)
	plt.show()

	return 0



if __name__== "__main__":
  main()
