import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Colores para los estados
GREY = (0.78, 0.78, 0.78)  # Susceptible
RED = (0.96, 0.15, 0.15)   # Infectado
GREEN = (0, 0.86, 0.03)    # Recuperado

# Update SIR_PARAMS
SIR_PARAMS = {
    "N": 200,
    "I0": 1,
    "infection_radius": 0.4,
    "infection_rate": 1,
    "recovery_rate": 0.1,
    "xmin": -2,          # Add boundary parameters
    "xmax": 2,
    "ymin": -2,
    "ymax": 2,
    "min_infection_time": 1
}

class SIRModelVisual:
    def __init__(self, params):
        # Basic parameters
        self.N = params["N"]
        self.I0 = params["I0"]
        
        # Add boundary parameters
        self.xmin = params.get("xmin", -2)
        self.xmax = params.get("xmax", 2)
        self.ymin = params.get("ymin", -2)
        self.ymax = params.get("ymax", 2)
        
        # Other parameters
        self.infection_radius = params["infection_radius"]
        self.recovery_rate = params["recovery_rate"]
        self.infection_rate = params.get("infection_rate", 0.3)
        self.min_infection_time = params.get("min_infection_time", 5)
        self.infection_durations = np.zeros(params["N"])
        
        # Tracking data
        self.S_data = []
        self.I_data = []
        self.R_data = []
        self.time_data = []
        self.day = 0
        
        # Initialize population
        self.df = self._initialize_population(params)
        self.setup_visualization()
        
    def _initialize_population(self, params):
        # Generar posiciones uniformes
        x = np.random.uniform(params["xmin"], params["xmax"], self.N)
        y = np.random.uniform(params["ymin"], params["ymax"], self.N)
        
        # Asignar estados iniciales
        states = np.array(['S'] * self.N)
        infected_idx = np.random.choice(self.N, self.I0, replace=False)
        states[infected_idx] = 'I'
        
        # Crear DataFrame
        df = pd.DataFrame({
            'id': range(self.N),
            'x': x,
            'y': y,
            'state': states,
            'iteration': 0
        })
        
        return df
    
    def check_distance(self, x1, y1, x2, y2):
        """Revisa si dos puntos están dentro del radio de infección"""
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance < self.infection_radius
    
    def update_states(self):
        """Actualiza estados basado en distancias y probabilidades"""
        # Obtener infectados y susceptibles
        infected = self.df[self.df['state'] == 'I']
        susceptible = self.df[self.df['state'] == 'S']
        
        # Update infection durations for infected individuals
        infected_mask = self.df['state'] == 'I'
        self.infection_durations[infected_mask] += 1
        
        # Check infections
        infected = self.df[infected_mask]
        susceptible = self.df[self.df['state'] == 'S']
        
        for _, sus in susceptible.iterrows():
            for _, inf in infected.iterrows():
                if self.check_distance(sus['x'], sus['y'], inf['x'], inf['y']):
                    if np.random.random() < self.infection_rate:  # Use infection_rate instead
                        self.df.loc[sus['id'], 'state'] = 'I'
                        break
        
        # Check recoveries only for those who met minimum infection time
        can_recover = (infected_mask) & (self.infection_durations >= self.min_infection_time)
        recovery_candidates = self.df[can_recover]
        
        for _, inf in recovery_candidates.iterrows():
            if np.random.random() < self.recovery_rate:
                self.df.loc[inf['id'], 'state'] = 'R'
                self.infection_durations[inf['id']] = 0  # Reset duration
        
        self.df['iteration'] = self.day

    def setup_visualization(self):
        # Create figure with two subplots side by side
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left subplot - Population scatter
        self.scatter = self.ax1.scatter(self.df['x'], self.df['y'], 
                                      c=self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
        self.ax1.set_xlim(SIR_PARAMS["xmin"], SIR_PARAMS["xmax"])
        self.ax1.set_ylim(SIR_PARAMS["ymin"], SIR_PARAMS["ymax"])
        
        # Right subplot - SIR curves
        self.ax2.set_xlim(0, 100)
        self.ax2.set_ylim(0, self.N)
        self.ax2.set_xlabel('Días')
        self.ax2.set_ylabel('Población')
        
        # Initialize curves
        self.line_s, = self.ax2.plot([], [], 'k-', label=f'Susceptibles: {self.N-self.I0}')
        self.line_i, = self.ax2.plot([], [], 'r-', label=f'Infectados: {self.I0}')
        self.line_r, = self.ax2.plot([], [], 'g-', label=f'Recuperados: 0')
        self.ax2.legend()
        
        plt.tight_layout()

    def update(self, frame):
        # Update states
        self.update_states()
        
        # Update scatter plot
        self.scatter.set_color(self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
        self.ax1.set_title(f"Día {self.day}")
        
        # Calculate counts once
        s_count = sum(self.df['state'] == 'S')
        i_count = sum(self.df['state'] == 'I')
        r_count = sum(self.df['state'] == 'R')
        
        # Update data for curves
        self.time_data.append(self.day)
        self.S_data.append(s_count)
        self.I_data.append(i_count)
        self.R_data.append(r_count)
        
        # Update lines
        self.line_s.set_data(self.time_data, self.S_data)
        self.line_i.set_data(self.time_data, self.I_data)
        self.line_r.set_data(self.time_data, self.R_data)
        
        # Update labels with current counts
        self.line_s.set_label(f'Susceptibles: {s_count}')
        self.line_i.set_label(f'Infectados: {i_count}')
        self.line_r.set_label(f'Recuperados: {r_count}')
        
        # Force legend update
        self.ax2.legend()
        
        # Adjust x axis limit if needed
        if self.day >= self.ax2.get_xlim()[1]:
            self.ax2.set_xlim(0, self.day * 1.5)
        
        self.day += 1

    def animate(self):
        self.anim = ani.FuncAnimation(self.fig, self.update, frames=100, interval=200, repeat=False)
        plt.show()
    
    def show_frames(self, frame_numbers):
        """Display specific frames of the simulation with both scatter and SIR curves"""
        # Save original data
        original_df = self.df.copy()
        original_time = self.time_data.copy()
        original_S = self.S_data.copy()
        original_I = self.I_data.copy()
        original_R = self.R_data.copy()
        original_day = self.day
        
        for frame in frame_numbers:
            # Reset simulation state
            self.df = original_df.copy()
            self.time_data = []
            self.S_data = []
            self.I_data = []
            self.R_data = []
            self.day = 0
            self.infection_durations = np.zeros(self.N)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Run simulation up to desired frame
            for _ in range(frame):
                self.update(None)
            
            # Left subplot - Population scatter
            ax1.scatter(self.df['x'], self.df['y'],
                       c=self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
            
            # Set equal aspect ratio and limits for scatter
            ax1.set_aspect('equal')
            ax1.set_xlim(self.xmin, self.xmax)
            ax1.set_ylim(self.ymin, self.ymax)
            ax1.set_title(f'Día {self.day}')
            
            # Right subplot - SIR curves
            ax2.set_xlim(0, max(100, frame))
            ax2.set_ylim(0, self.N)
            ax2.set_xlabel('Días')
            ax2.set_ylabel('Población')
            
            # Plot SIR curves up to current day
            ax2.plot(self.time_data, self.S_data, 'k-', label=f'Susceptibles: {self.S_data[-1]}')
            ax2.plot(self.time_data, self.I_data, 'r-', label=f'Infectados: {self.I_data[-1]}')
            ax2.plot(self.time_data, self.R_data, 'g-', label=f'Recuperados: {self.R_data[-1]}')
            ax2.legend()
            ax2.set_title('Curvas SIR')
            
            plt.tight_layout()
            plt.show()
        
        # Restore original state
        self.df = original_df
        self.time_data = original_time
        self.S_data = original_S
        self.I_data = original_I
        self.R_data = original_R
        self.day = original_day

def main():
    sir_visual = SIRModelVisual(SIR_PARAMS)
    # sir_visual.animate()
    sir_visual.show_frames([5,20,40])

if __name__ == "__main__":
    main()
