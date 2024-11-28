import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani


GREY = (0.78, 0.78, 0.78)  # Susceptible
RED = (0.96, 0.15, 0.15)   # Infectado
GREEN = (0, 0.86, 0.03)    # Recuperado


SIR_PARAMS = {
    "N": 200,              
    "I0": 1,             
    "infection_radius": 1, 
    "infection_rate": 0.1,   
    "recovery_rate": 0.1,   
    "diameter": 9, 
    "min_infection_time": 1  
}

class SIRModelVisual:
    def __init__(self, params):
        self.N = params["N"]
        self.I0 = params["I0"]
        self.diameter = params.get("diameter", 4)
        self.radius = self.diameter / 2
        
        self.infection_radius = params["infection_radius"]
        self.recovery_rate = params["recovery_rate"]
        self.infection_rate = params.get("infection_rate", 0.3)
        self.min_infection_time = params.get("min_infection_time", 5)
        self.infection_durations = np.zeros(params["N"])
        
        self.S_data = []
        self.I_data = []
        self.R_data = []
        self.time_data = []
        self.day = 0
        

        self.df = self._initialize_population(params)
        self.setup_visualization()

    def _initialize_population(self, params):

        N = self.N
        R = self.radius
        
        points = []
        while len(points) < N:

            a = np.random.random() * 2 * np.pi
            r = R * np.sqrt(np.random.random())
            

            jitter = 0.1 * R  
            x = r * np.cos(a) + np.random.uniform(-jitter, jitter)
            y = r * np.sin(a) + np.random.uniform(-jitter, jitter)
            
            if x*x + y*y <= R*R:
                points.append((x, y))
        
        points = np.array(points)

        states = np.array(['S'] * N)
        infected_idx = np.random.choice(N, self.I0, replace=False)
        states[infected_idx] = 'I'
        
        return pd.DataFrame({
            'id': range(N),
            'x': points[:, 0],
            'y': points[:, 1],
            'state': states,
            'iteration': 0
        })

    def check_distance(self, x1, y1, x2, y2):
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance < self.infection_radius
    
    def update_states(self):

        infected = self.df[self.df['state'] == 'I']
        susceptible = self.df[self.df['state'] == 'S']
        
        infected_mask = self.df['state'] == 'I'
        self.infection_durations[infected_mask] += 1
        
        infected = self.df[infected_mask]
        susceptible = self.df[self.df['state'] == 'S']
        
        for _, sus in susceptible.iterrows():
            for _, inf in infected.iterrows():
                if self.check_distance(sus['x'], sus['y'], inf['x'], inf['y']):
                    if np.random.random() < self.infection_rate:  
                        self.df.loc[sus['id'], 'state'] = 'I'
                        break
        
        can_recover = (infected_mask) & (self.infection_durations >= self.min_infection_time)
        recovery_candidates = self.df[can_recover]
        
        for _, inf in recovery_candidates.iterrows():
            if np.random.random() < self.recovery_rate:
                self.df.loc[inf['id'], 'state'] = 'R'
                self.infection_durations[inf['id']] = 0 
        
        self.df['iteration'] = self.day

    def setup_visualization(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        self.scatter = self.ax1.scatter(self.df['x'], self.df['y'], 
                                      c=self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
        
        circle = plt.Circle((0, 0), self.radius, fill=False, color='black')
        self.ax1.add_artist(circle)
        
        self.ax1.set_aspect('equal')
        limit = self.radius * 1.1  
        self.ax1.set_xlim(-limit, limit)
        self.ax1.set_ylim(-limit, limit)
        
        self.ax2.set_xlim(0, 100)
        self.ax2.set_ylim(0, self.N)
        self.ax2.set_xlabel('Días')
        self.ax2.set_ylabel('Población')
        
        self.line_s, = self.ax2.plot([], [], 'k-', label=f'Susceptibles: {self.N-self.I0}')
        self.line_i, = self.ax2.plot([], [], 'r-', label=f'Infectados: {self.I0}')
        self.line_r, = self.ax2.plot([], [], 'g-', label=f'Recuperados: 0')
        self.ax2.legend()
        
        plt.tight_layout()

    def update(self, frame):

        self.update_states()
        

        self.scatter.set_color(self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
        self.ax1.set_title(f"Día {self.day}")
        

        s_count = sum(self.df['state'] == 'S')
        i_count = sum(self.df['state'] == 'I')
        r_count = sum(self.df['state'] == 'R')
        
        self.line_s.set_label(f'Susceptibles: {s_count}')
        self.line_i.set_label(f'Infectados: {i_count}')
        self.line_r.set_label(f'Recuperados: {r_count}')
        
        self.ax2.legend()
        
        self.time_data.append(self.day)
        self.S_data.append(s_count)
        self.I_data.append(i_count)
        self.R_data.append(r_count)
        

        self.line_s.set_data(self.time_data, self.S_data)
        self.line_i.set_data(self.time_data, self.I_data)
        self.line_r.set_data(self.time_data, self.R_data)
        

        if self.day >= self.ax2.get_xlim()[1]:
            self.ax2.set_xlim(0, self.day * 1.5)
        
        self.day += 1

    def animate(self):
        self.anim = ani.FuncAnimation(self.fig, self.update, frames=100, interval=200, repeat=False)
        self.anim.save('Animaciones/sir_simulation_circulo.mp4', writer='ffmpeg')
        plt.show()

    def show_frames(self, frame_numbers):
        original_df = self.df.copy()
        original_time = self.time_data.copy()
        original_S = self.S_data.copy()
        original_I = self.I_data.copy()
        original_R = self.R_data.copy()
        original_day = self.day
        
        for frame in frame_numbers:

            self.df = original_df.copy()
            self.time_data = [0]  
            self.S_data = [self.N - self.I0]  
            self.I_data = [self.I0]  
            self.R_data = [0]  
            self.day = 0
            self.infection_durations = np.zeros(self.N)
            

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            

            for _ in range(frame):
                self.update(None)
            
            ax1.scatter(self.df['x'], self.df['y'],
                       c=self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
            

            circle = plt.Circle((0, 0), self.radius, fill=False, color='black')
            ax1.add_artist(circle)
            

            ax1.set_aspect('equal')
            limit = self.radius * 1.1  
            ax1.set_xlim(-limit, limit)
            ax1.set_ylim(-limit, limit)
            ax1.set_title(f'Día {self.day}')
            

            ax2.set_xlim(0, max(100, self.day))
            ax2.set_ylim(0, self.N)
            ax2.set_xlabel('Días')
            ax2.set_ylabel('Población')
            

            current_s = self.S_data[-1] if self.S_data else 0
            current_i = self.I_data[-1] if self.I_data else 0
            current_r = self.R_data[-1] if self.R_data else 0
            
            ax2.plot(self.time_data, self.S_data, 'k-', label=f'Susceptibles: {current_s}')
            ax2.plot(self.time_data, self.I_data, 'r-', label=f'Infectados: {current_i}')
            ax2.plot(self.time_data, self.R_data, 'g-', label=f'Recuperados: {current_r}')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
        
        self.df = original_df
        self.time_data = original_time
        self.S_data = original_S
        self.I_data = original_I
        self.R_data = original_R
        self.day = original_day

def main():
    sir_visual = SIRModelVisual(SIR_PARAMS)
    sir_visual.animate() #Comentar para ver solo los frames
    #sir_visual.show_frames([5,20,40]) #comentar para ver la animación

if __name__ == "__main__":
    main()