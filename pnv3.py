import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import numpy.linalg as la
from simple_pid import PID

# calculation constants
CONST_SIMTIME = 60 #s
CONST_DT = 0.2 #s
N = int(CONST_SIMTIME / CONST_DT) # number of frames

# physical missile specifications
MISSILE_MASS = 700 #kg
MISSILE_LENGTH = 5.5 #m
MISSILE_RADIUS = 0.2 #m

# aerodynamic missile specifications
MISSILE_CP_DIST = 0.2 #m 
MISSILE_CT_SA = 0.0175 #m^2
MISSILE_CS_DIST = 2.5 #m

# air density calculation as a function of altitude
def air_density(altitude):
    return 1.225 * (1 - (0.00002255691 * altitude))**4.2561

class Missile:
    def __init__(self, ipos:list, ivel:list):
        # MISSILE PHYSICS ARRAYS
        self.pos = np.add(np.zeros((N+1, 3)), np.array(ipos)) # position

        self.vel = np.add(np.zeros((N+1, 3)), np.array(ivel)) # velocity

        self.fsum = np.zeros((N+1, 3)) # sum of forces

        self.rcfs = np.zeros((N+1, 3)) # sum of control surface forces relative to control surface

        self.acfs = np.zeros((N+1, 3)) # sum of control surface forces relative to missile

        self.spd = np.zeros(N+1) # speed

        self.aoa = np.zeros(N+1) # angle of attack

        self.rho = np.zeros(N+1) # air density at altitude

        self.los = np.zeros(N+1) # line of sight to target

        self.clv = np.zeros(N+1) # closing velocity to target

        self.axd = np.zeros(N+1) # axial body drag

        self.nmd = np.zeros(N+1) # normal body drag

        self.surfang = np.zeros(N+1) # control surface angle

        self.axc = np.zeros(N+1) # control surface axial drag

        self.nmc = np.zeros(N+1) # control surface normal drag

        self.thr = np.zeros(N+1) # missile thrust

        self.rsm = np.zeros(N+1) # restoring moment

        self.ydm = np.zeros(N+1) # yaw damping moment

        self.csm = np.zeros(N+1) # control surface moment

    def update(self, n, surf_input):
        # COMMON PROPERTIES

        # angle of attack
        self.aoa[n] = self.pos[n, 2] - np.arctan2(self.vel[n, 1], self.vel[n, 0])

        # magnitude of velocity vector (speed)
        self.spd[n] = np.sqrt(self.vel[n, 0]**2 + self.vel[n, 1]**2)

        # air density
        self.rho[n] = 1.225 * (1 - (0.00002255691 * self.pos[n, 1]))**4.2561

        # Reynolds number
        # self.re = ((self.spd[n]**2) * MISSILE_RADIUS * 2) / 1.48e-5

        # Mach number
        # self.ma = self.spd[n] / (340.3 - 0.004 * self.pos[n, 1])

        # AERODYNAMIC FORCES
        # MISSILE BODY DRAG
        # axial drag
        self.kda = 0.5 * np.cos(self.aoa[n])
        self.axd[n] = self.rho[n] * ((2 * MISSILE_RADIUS)**2) * (self.spd[n]**2) * self.kda

        # normal drag
        self.kn = 1.4 * np.sin(self.aoa[n])
        self.nmd[n] = self.rho[n] * ((2 * MISSILE_RADIUS)**2) * (self.spd[n]**2) * self.kn

        # restoring moment
        self.rsm[n] = self.nmd[n] * MISSILE_CP_DIST

        # CONTROL SURFACE DRAG
        # control surface angle
        self.surfang[n] = surf_input

        # axial control surface drag
        self.ksa = 0.05 * np.cos(self.aoa[n] + self.surfang[n])
        self.axc[n] = self.rho[n] * (MISSILE_CT_SA) * (self.spd[n]**2) * self.ksa

        # normal control surface drag
        self.ksn = 1.5 * np.sin(self.aoa[n] + self.surfang[n])
        self.nmc[n] = self.rho[n] * (MISSILE_CT_SA) * (self.spd[n]**2) * self.ksn

        # control surface moment
        self.csm[n] = -self.nmc[n] * MISSILE_CS_DIST

        # control surface force summation
        self.rcfs[n] = np.array([-self.axc[n], self.nmc[n], self.csm[n]])

        # rotation matrix for vector rotation
        c = np.cos(self.surfang[n])
        s = np.sin(self.surfang[n])
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # absolute control surface force
        self.acfs[n] = np.dot(r, self.rcfs[n])

        # yaw damping
        self.km = 500
        self.ydm[n] = self.rho[n] * ((2 * MISSILE_RADIUS)**4) * self.vel[n, 2] * self.spd[n] * self.km

        # missile thrust
        self.thr[n] = MISSILE_MASS * 40 

        # force summation
        self.fsum[n] = np.array([(-self.axd[n] + self.thr[n]), (self.nmd[n]), (-self.rsm[n] - self.ydm[n])]) + self.acfs[n]

        # rotation matrix for vector rotation
        c = np.cos(self.pos[n, 2])
        s = np.sin(self.pos[n, 2])
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # rotation of force sum array and division by mass
        self.absaccel = np.divide(np.dot(r, self.fsum[n]), np.full(3, MISSILE_MASS))

        # adds missile total acceleration to velocity and adds gravity
        self.vel[n+1] = self.vel[n] + ((self.absaccel + np.array([0, -9.81, 0])) * CONST_DT)
        self.pos[n+1] = self.pos[n] + self.vel[n] * CONST_DT

    def purepursuit(self, n, target):
        self.los[n] = np.arctan2(target.pos[n, 1] - self.pos[n, 1], target.pos[n, 0] - self.pos[n, 0]) - self.pos[n, 2]
        output = (self.los[n] * -400)/(self.spd[n-1]**2)

        return max(-np.pi/4, min(output, np.pi/4))

    def pronav(self, n, target):
        # line of sight from missile to target
        # self.los[n] = np.arctan2(target.pos[n, 1] - self.pos[n, 1], target.pos[n, 0] - self.pos[n, 0]) - self.pos[n, 2]

        # line of sight from missile velocity vector to target
        self.los[n] = np.arctan2(target.pos[n, 1] - self.pos[n, 1], target.pos[n, 0] - self.pos[n, 0]) - np.arctan2(self.vel[n, 1], self.vel[n, 0])
        
        # calculation of closing velocity
        dp = (target.pos[n] - self.pos[n])[:-1]
        dv = (target.vel[n] - self.vel[n])[:-1] 
        self.clv[n] = -(np.dot(dv, dp)/la.norm(dp))

        dlos = (self.los[n] - self.los[n-1]) / CONST_DT

        desired_acceleration = 4 * ((self.los[n] - self.los[n-1]) / CONST_DT) * self.clv[n]
        current_acceleration = self.fsum[n-1, 1] / MISSILE_MASS

        # reverse mathematics to find the control surface angle that will produce the desired acceleration

        # normal drag
        desired_aoa = np.arcsin(max(-1, min((desired_acceleration * MISSILE_MASS) / (self.rho[n-1] * ((2 * MISSILE_RADIUS)**2) * (self.spd[n-1]**2) * 1.4), 1))) 

        pid = PID(-0.5, 0, 100, setpoint=desired_aoa)
        output = pid(self.aoa[n])
        print(desired_aoa) 

        # print((self.los[n] - self.los[n-1]) / CONST_DT)
        # print(self.clv[n])
        # print("desired = " + str(desired_acceleration) + " current = " + str(current_acceleration) + " output = " + str(output))

        return max(-np.pi/4, min(output, np.pi/4))

class Aircraft:
    def __init__(self, ipos:list, ivel:list):
        self.pos = np.add(np.zeros((N+1, 3)), np.array(ipos)) # position
        self.vel = np.add(np.zeros((N+1, 3)), np.array(ivel)) # velocity

    def update(self, n):
        self.vel[n+1] = self.vel[n]
        self.pos[n+1] = self.pos[n] + self.vel[n] * CONST_DT

XLIM = 42000
YLIM = 25000

missile = Missile([-3000, 0, np.pi/4], [10, 10, 0])
target = Aircraft([0, 17000, 0], [100, 0, 0])

# plot
fig, axs = plt.subplots(4)

mtrack = axs[0].plot(0, 0, label="Missile")[0]
ttrack = axs[0].plot(0, 0, label="Target")[0]
nmdtrack = axs[3].plot(0, 0, label="Normal drag force")[0]
spdtrack = axs[2].plot(0, 0, label="Missile speed")[0]
aoatrack = axs[1].plot(0, 0, label="Angle of attack")[0]


def update(frame):
    # missile update function
    missile.update(frame, missile.pronav(frame, target))
    # missile.update(frame, missile.purepursuit(frame, target))
    target.update(frame)
    # missile.pronav(frame, target)

    # missile position plot
    mtrack.set_xdata([x for x, y, z in missile.pos[:frame]])
    mtrack.set_ydata([y for x, y, z in missile.pos[:frame]])

    # target position plot
    ttrack.set_xdata([x for x, y, z in target.pos[:frame]])
    ttrack.set_ydata([y for x, y, z in target.pos[:frame]])

    # missile speed plot
    spdtrack.set_xdata([x for x, y, z in missile.pos[:frame]])
    spdtrack.set_ydata([x for x in missile.spd[:frame]])
    
    # missile angle of attack plot
    aoatrack.set_xdata([x for x, y, z in missile.pos[:frame]])
    aoatrack.set_ydata([x for x in missile.aoa[:frame]])

    # missile normal drag plot
    nmdtrack.set_xdata([x for x, y, z in missile.pos[:frame]])
    nmdtrack.set_ydata([y / MISSILE_MASS for x, y, z in missile.fsum[:frame]])

    return(mtrack, ttrack)

axs[0].set(xlim=(-XLIM, XLIM), ylim=(0, YLIM), xlabel = "Position (m)", ylabel = "Altitude (m)")
axs[3].set(xlim=(-XLIM, XLIM), ylim=(-50, 50), xlabel = "Position (m)", ylabel = "Drag forces n to missile (N)")
axs[2].set(xlim=(-XLIM, XLIM), ylim=(0, 1000), xlabel = "Position (m)", ylabel = "Speed (m/s)")
axs[1].set(xlim=(-XLIM, XLIM), ylim=(-np.pi, np.pi), xlabel = "Position (m)", ylabel = "Angle of attack (rad)")
axs[1].legend()



ani = ani.FuncAnimation(fig=fig, func=update, frames=N, interval=10)
ani.save("test.mp4")

plt.savefig("pnv3.png")
plt.show()

