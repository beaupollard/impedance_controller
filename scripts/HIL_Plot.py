import numpy as np
import matplotlib.pyplot as plt


gt=np.loadtxt('ground_truthv3.txt')
im=np.loadtxt('Impedancev3.txt')
dt=0.005
t=np.linspace(0,dt*len(gt),len(gt))

font = {'font.serif' : 'Times New Toman',
        'font.size'   : 18}
plt.rcParams.update(font)


plt.plot(t,gt,'r',linewidth=4)
plt.plot(t,im,'b',linewidth=4)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend(['Ground Truth','Impedance Matched'])
plt.savefig('Impedancev2.png',dpi=300,bbox_inches='tight')
# plt.show()