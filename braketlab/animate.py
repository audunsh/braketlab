from matplotlib.animation import FuncAnimation
import bubblebox as bb
import numpy as np
import matplotlib.pyplot as plt


class animated_system():
    def __init__(self, system = None, n_steps_per_vis = 5, interval = 1):
        self.n_steps_per_vis = n_steps_per_vis
        self.system = system
        figsize = (6,4)
        
    
        plt.rcParams["figure.figsize"] = figsize

        
        self.fig, self.ax = plt.subplots()
        self.col = bb.colorscheme()

        self.scatterplot = False
        


        
        self.ani = FuncAnimation(self.fig, self.update, interval=interval, 
                                          init_func=self.setup_plot, blit=True,cache_frame_data=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        
        
        x, p = self.system.x, self.system.p
        w = self.system.w
        
        c1,c2,c3 = bb.colorscheme().getcol(np.random.uniform(0,1,3))
        
        self.bubbles = self.ax.plot(x, p, "-", color = (0,0,0), markersize = 1, label = "$\\vert \Psi \\vert^2$")[0]
        
        self.wf_imag = self.ax.plot(x, w.imag, "-", color = c1**.5, markersize = 1, label = "Im($\Psi$)")[0]
        self.wf_real = self.ax.plot(x, w.real, "-", color = c2**.5, markersize = 1, label = "Re($\Psi$)")[0]
        
        self.ax.legend()
        


        self.ax.axis([x[0], x[-1], -self.system.h, self.system.h])
        
        
        return self.bubbles,


        
        
    #@nb.jit
    def update(self, i):

        for i in range(self.n_steps_per_vis):
            self.system.advance()
            
        x, p = self.system.x, self.system.p
        w = self.system.w

        self.bubbles.set_data(x,p)
        self.wf_real.set_data(x,w.imag)
        self.wf_imag.set_data(x,w.real)

        return self.bubbles,
    
class system():
    def __init__(self, psi, x, t = 0, dt = 0.001, h = 1):
        self.psi = psi
        self.x = x
        self.t = t
        self.dt = dt
        self.h = h
        self.advance()
        
        
        
    def advance(self):
        """
        advance solution in time
        note: In order to make it optimal I used vectorized numpy,
              BUT the linear code is left commented out for clarity
        """
            
        w = self.psi(self.x, t=self.t)

        self.w = w
        self.p = (w.conj()*w).real
        self.t += self.dt
        
    def run(self,n_steps_per_vis=1, interval = 10):
        run_system = animated_system(system = self, n_steps_per_vis=n_steps_per_vis, interval = interval)
        plt.show()