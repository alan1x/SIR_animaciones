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
    "infection_radius": 0.4,
    "infection_rate": 1,
    "recovery_rate": 0.1,
    "xmin": -2,          
    "xmax": 2,
    "ymin": -2,
    "ymax": 2,
    "min_infection_time": 1
}

class SIRModelVisual:
    def __init__(self, params):

        self.N = params["N"]
        self.I0 = params["I0"]
        

        self.xmin = params.get("xmin", -2)
        self.xmax = params.get("xmax", 2)
        self.ymin = params.get("ymin", -2)
        self.ymax = params.get("ymax", 2)
        

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

        data = {
            'id': np.arange(params["N"]),
            'x': np.random.uniform(self.xmin, self.xmax, params["N"]),
            'y': np.random.uniform(self.ymin, self.ymax, params["N"]),
            'state': ['S'] * params["N"]
        }
        df = pd.DataFrame(data)
        initial_infected = np.random.choice(df.index, params["I0"], replace=False)
        df.loc[initial_infected, 'state'] = 'I'
        return df

    def check_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= self.infection_radius

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
        self.ax1.set_xlim(self.xmin, self.xmax)
        self.ax1.set_ylim(self.ymin, self.ymax)
        

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
        

        self.time_data.append(self.day)
        self.S_data.append(s_count)
        self.I_data.append(i_count)
        self.R_data.append(r_count)
        
        self.line_s.set_data(self.time_data, self.S_data)
        self.line_i.set_data(self.time_data, self.I_data)
        self.line_r.set_data(self.time_data, self.R_data)

        self.line_s.set_label(f'Susceptibles: {s_count}')
        self.line_i.set_label(f'Infectados: {i_count}')
        self.line_r.set_label(f'Recuperados: {r_count}')
        

        self.ax2.legend()
        

        if self.day >= self.ax2.get_xlim()[1]:
            self.ax2.set_xlim(0, self.day * 1.5)
        
        self.day += 1

    def animate(self):
        self.anim = ani.FuncAnimation(self.fig, self.update, frames=100, interval=200, repeat=False)
        self.anim.save('fase3/sir_simulation_rectangulo.mp4', writer='ffmpeg') #comentar para ver la animación y descomentar para guardarla
        plt.show()

    def show_frames(self, frame_numbers):

        original_df = self.df.copy()
        original_time = self.time_data.copy()
        original_S = self.S_data.copy()
        original_I = self.I_data.copy()
        original_R = self.R_data.copy()
        original_day = self.day
        
        for frame in frame_numbers:
            self.update(frame)
            self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            

            ax1.scatter(self.df['x'], self.df['y'], 
                        c=self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
            ax1.set_xlim(self.xmin, self.xmax)
            ax1.set_ylim(self.ymin, self.ymax)
            ax1.set_title(f'Día {self.day}')
            

            ax2.set_xlim(0, max(100, frame))
            ax2.set_ylim(0, self.N)
            ax2.set_xlabel('Días')
            ax2.set_ylabel('Población')
            
            ax2.plot(self.time_data, self.S_data, 'k-', label=f'Susceptibles: {self.S_data[-1]}')
            ax2.plot(self.time_data, self.I_data, 'r-', label=f'Infectados: {self.I_data[-1]}')
            ax2.plot(self.time_data, self.R_data, 'g-', label=f'Recuperados: {self.R_data[-1]}')
            ax2.legend()
            ax2.set_title('Curvas SIR')
            
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
    sir_visual.animate() 
    #sir_visual.show_frames([5,20,40]) #comentar para ver la animación

if __name__ == "__main__":
    main()
