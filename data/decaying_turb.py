import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np

lattice = lt.Lattice(lt.D2Q9, device = "cuda", use_native=False)
flow = lt.DecayingTurbulence(resolution=256, reynolds_number=10000, mach_number=0.05, lattice=lattice)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
energyspectrum = lt.EnergySpectrum(lattice, flow)
reporter = lt.ObservableReporter(energyspectrum, interval=500, out=None)
simulation.reporters.append(reporter)

u = lattice.convert_to_numpy(flow.units.convert_velocity_to_pu(lattice.u(simulation.f)))
u_norm = np.linalg.norm(u,axis=0)
plt.imshow(u_norm)
plt.title('Initialized velocity')
plt.show()