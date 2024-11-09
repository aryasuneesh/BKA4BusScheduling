import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class NetworkParams:
    num_cities: int
    fleet_size: int
    bus_capacity: int
    time_slots: int
    peak_hours: List[Tuple[int, int]]
    base_demand: Dict[str, int]
    peak_multiplier: float
    cost_per_km: float

class BusScheduleGenerator:
    def __init__(self, params: NetworkParams):
        self.params = params
        self.distance_matrix = None
        self.base_demand_matrix = None
        self.time_demand_matrix = None
        
    def generate_distance_matrix(self) -> np.ndarray:
        """Generate random but realistic distances between cities"""
        n = self.params.num_cities
        # Generate random distances between 50-500 km
        distances = np.random.uniform(50, 500, (n, n))
        # Make matrix symmetric
        distances = (distances + distances.T) / 2
        # Set diagonal to 0
        np.fill_diagonal(distances, 0)
        self.distance_matrix = distances
        return distances
    
    def generate_base_demand(self) -> np.ndarray:
        """Generate base demand between city pairs"""
        n = self.params.num_cities
        demand_levels = list(self.params.base_demand.values())
        # Randomly assign demand levels to city pairs
        demands = np.random.choice(demand_levels, (n, n))
        # Make matrix symmetric and set diagonal to 0
        demands = (demands + demands.T) / 2
        np.fill_diagonal(demands, 0)
        self.base_demand_matrix = demands
        return demands
        
    def generate_time_demand(self) -> np.ndarray:
        """Generate time-dependent demand matrix"""
        n = self.params.num_cities
        t = self.params.time_slots
        
        # Initialize 3D matrix (cities × cities × time_slots)
        demand = np.zeros((n, n, t))
        
        # Fill with base demand
        for time in range(t):
            demand[:, :, time] = self.base_demand_matrix
            
            # Apply peak multiplier during peak hours
            for peak_start, peak_end in self.params.peak_hours:
                if peak_start <= time < peak_end:
                    demand[:, :, time] *= self.params.peak_multiplier
                    
        self.time_demand_matrix = demand
        return demand

    def generate_initial_population(self, pop_size: int) -> np.ndarray:
        """Generate initial population of bus schedules"""
        n = self.params.num_cities
        t = self.params.time_slots
        fleet = self.params.fleet_size
        
        # Each solution is a 3D matrix (buses × cities × time_slots)
        # Value 1 indicates bus i is in city j at time t
        population = []
        
        for _ in range(pop_size):
            # Initialize empty schedule
            schedule = np.zeros((fleet, n, t))
            
            # For each bus, generate random initial route
            for bus in range(fleet):
                current_city = np.random.randint(0, n)
                for time in range(t):
                    schedule[bus, current_city, time] = 1
                    # Randomly move to adjacent city with small probability
                    if np.random.random() < 0.2:
                        possible_cities = np.where(self.distance_matrix[current_city] > 0)[0]
                        if len(possible_cities) > 0:
                            current_city = np.random.choice(possible_cities)
            
            population.append(schedule)
        
        return np.array(population)
    
    def is_feasible_schedule(self, schedule: np.ndarray) -> bool:
        """Check if schedule meets basic constraints"""
        # Check if each bus is in exactly one city at each time
        for t in range(self.params.time_slots):
            for bus in range(self.params.fleet_size):
                if not np.sum(schedule[bus, :, t]) == 1:
                    return False
        
        # Check if bus movements are physically possible
        for t in range(1, self.params.time_slots):
            for bus in range(self.params.fleet_size):
                prev_city = np.where(schedule[bus, :, t-1] == 1)[0][0]
                curr_city = np.where(schedule[bus, :, t] == 1)[0][0]
                if prev_city != curr_city:
                    # Check if movement is possible within one time slot
                    if self.distance_matrix[prev_city, curr_city] > 100:  # Assuming 100km/hour max speed
                        return False
        
        return True
    
    def generate_dataset(self, pop_size: int = 30):
        """Generate complete dataset including multiple populations"""
        self.generate_distance_matrix()
        self.generate_base_demand()
        self.generate_time_demand()
        initial_population = self.generate_initial_population(pop_size)
        
        return {
            'distance_matrix': self.distance_matrix,
            'time_demand_matrix': self.time_demand_matrix,
            'initial_population': initial_population,
            'params': self.params
        }