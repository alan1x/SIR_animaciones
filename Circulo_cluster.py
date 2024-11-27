import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Colores para los estados
GREY = (0.78, 0.78, 0.78)  # Susceptible
RED = (0.96, 0.15, 0.15)   # Infectado
GREEN = (0, 0.86, 0.03)    # Recuperado

# Parámetros del modelo SIR
SIR_PARAMS = {
    "N": 200,
    "I0": 1,
    "infection_radius": 0.4,
    "infection_rate": 1,
    "recovery_rate": 0.1,
    "diameter": 4,
    "min_infection_time": 1,
    # Cluster configuration
    "cluster": {
        "x": 0,        # Cluster center x 
        "y": 0,        # Cluster center y
        "radius": 1,   # Cluster radius
        "density": 0.7,  # Proportion of population in cluster
        "sigma": 1     # Standard deviation for normal distribution
    }
}

class SIRModelVisual:
    def __init__(self, params):
        # Basic parameters
        self.N = params["N"]
        self.I0 = params["I0"]
        self.diameter = params.get("diameter", 4)
        self.radius = self.diameter / 2
        
        # Other parameters remain the same...
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
        N = self.N
        R = self.radius
        cluster = params.get("cluster", {})
        
        points = []
        cluster_size = int(N * cluster.get("density", 0.3))
        regular_size = N - cluster_size
        
        # Generate cluster points with normal distribution
        cluster_center = np.array([cluster.get("x", 0), cluster.get("y", 0)])
        cluster_radius = cluster.get("radius", 0.5)
        sigma = cluster.get("sigma", 0.2)
        
        while len(points) < cluster_size:
            # Generate points with normal distribution around cluster center
            point = np.random.normal(loc=cluster_center, scale=sigma, size=2)
            x, y = point
            
            # Check if point is within main circle and cluster radius
            if x*x + y*y <= R*R and ((x-cluster_center[0])**2 + (y-cluster_center[1])**2 <= cluster_radius**2):
                points.append((x, y))
        
        # Generate remaining points uniformly in main circle
        while len(points) < N:
            a = np.random.random() * 2 * np.pi
            r = R * np.sqrt(np.random.random())
            x = r * np.cos(a)
            y = r * np.sin(a)
            
            # Avoid cluster area
            dx = x - cluster_center[0]
            dy = y - cluster_center[1]
            if dx*dx + dy*dy > cluster_radius*cluster_radius:
                points.append((x, y))
        
        points = np.array(points)
        
        # Assign initial states
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
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left subplot - Population with cluster
        self.scatter = self.ax1.scatter(self.df['x'], self.df['y'], 
                                      c=self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
        
        # Add main boundary circle
        circle = plt.Circle((0, 0), self.radius, fill=False, color='black')
        self.ax1.add_artist(circle)
        
        # Add dashed cluster boundary
        cluster = SIR_PARAMS["cluster"]
        cluster_circle = plt.Circle((cluster["x"], cluster["y"]), 
                                  cluster["radius"],
                                  fill=False, 
                                  color='black', 
                                  linestyle='--',
                                  alpha=0.5)
        self.ax1.add_artist(cluster_circle)
        
        # Set equal aspect ratio and limits
        self.ax1.set_aspect('equal')
        limit = self.radius * 1.1  # Add 10% margin
        self.ax1.set_xlim(-limit, limit)
        self.ax1.set_ylim(-limit, limit)
        
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
        
        # Add text annotations for counts
        self.count_text = self.ax1.text(
            -limit * 0.9, limit * 0.9,
            '',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        
        # Add text annotations for counts in the SIR curves plot
        self.count_text = self.ax2.text(
            0.02, 0.98,  # Position in axes coordinates (top-left)
            '',
            transform=self.ax2.transAxes,  # Use axes coordinates
            fontsize=9,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    def update(self, frame):
        # Update states
        self.update_states()
        
        # Update scatter plot
        self.scatter.set_color(self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
        self.ax1.set_title(f"Día {self.day}")
        
        # Count current states
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
        self.line_s.set_label(f'Susceptibles: {s_count}')
        self.line_i.set_label(f'Infectados: {i_count}')
        self.line_r.set_label(f'Recuperados: {r_count}')
        self.ax2.legend()
        
        # Adjust x axis limit if needed
        if self.day >= self.ax2.get_xlim()[1]:
            self.ax2.set_xlim(0, self.day * 1.5)
        
        self.day += 1

    def animate(self):
        self.anim = ani.FuncAnimation(self.fig, self.update, frames=100, interval=200, repeat=False)
        plt.show()

    def show_frames(self, frame_numbers):
        """Display specific frames of the simulation with cluster visualization"""
        # Save original state
        original_df = self.df.copy()
        original_time = self.time_data.copy()
        original_S = self.S_data.copy()
        original_I = self.I_data.copy()
        original_R = self.R_data.copy()
        original_day = self.day
        
        for frame in frame_numbers:
            # Reset simulation state
            self.df = original_df.copy()
            self.time_data = [0]  # Initialize with frame 0
            self.S_data = [self.N - self.I0]  # Initial susceptible count
            self.I_data = [self.I0]  # Initial infected count
            self.R_data = [0]  # Initial recovered count
            self.day = 0
            self.infection_durations = np.zeros(self.N)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Run simulation up to desired frame
            for _ in range(frame):
                self.update(None)
            
            # Left subplot - Population with cluster
            ax1.scatter(self.df['x'], self.df['y'],
                       c=self.df['state'].map({'S': GREY, 'I': RED, 'R': GREEN}))
            
            # Add main circle boundary
            circle = plt.Circle((0, 0), self.radius, fill=False, color='black')
            ax1.add_artist(circle)
            
            # Add cluster boundary
            cluster = SIR_PARAMS["cluster"]
            cluster_circle = plt.Circle((cluster["x"], cluster["y"]), 
                                      cluster["radius"],
                                      fill=False, 
                                      color='black', 
                                      linestyle='--',
                                      alpha=0.5)
            ax1.add_artist(cluster_circle)
            
            # Set equal aspect ratio and limits
            ax1.set_aspect('equal')
            limit = self.radius * 1.1  # Add 10% margin
            ax1.set_xlim(-limit, limit)
            ax1.set_ylim(-limit, limit)
            ax1.set_title(f'Día {self.day}')
            
            # Right subplot - SIR curves
            ax2.set_xlim(0, max(100, self.day))
            ax2.set_ylim(0, self.N)
            ax2.set_xlabel('Días')
            ax2.set_ylabel('Población')
            
            # Plot SIR curves with safe access to last values
            current_s = self.S_data[-1] if self.S_data else 0
            current_i = self.I_data[-1] if self.I_data else 0
            current_r = self.R_data[-1] if self.R_data else 0
            
            ax2.plot(self.time_data, self.S_data, 'k-', label=f'Susceptibles: {current_s}')
            ax2.plot(self.time_data, self.I_data, 'r-', label=f'Infectados: {current_i}')
            ax2.plot(self.time_data, self.R_data, 'g-', label=f'Recuperados: {current_r}')
            ax2.legend()
            
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
    sir_visual.animate()
    # sir_visual.show_frames([5, 20, 40])  # Show days 0, 10, and 20

if __name__ == "__main__":
    main()
