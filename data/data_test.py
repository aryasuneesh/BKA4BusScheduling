import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from synthetic_data_generator import NetworkParams, BusScheduleGenerator

def test_data_generation():
    """Test synthetic data generation and visualize results"""
    
    # Initialize network parameters
    params = NetworkParams(
        num_cities=5,
        fleet_size=20,
        bus_capacity=50,
        time_slots=24,
        peak_hours=[(7,9), (17,19)],
        base_demand={'low': 10, 'medium': 20, 'high': 30},
        peak_multiplier=2.0,
        cost_per_km=10
    )
    
    # Create generator instance
    generator = BusScheduleGenerator(params)
    
    # Generate complete dataset
    dataset = generator.generate_dataset(pop_size=30)
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Number of cities: {params.num_cities}")
    print(f"Population size: {len(dataset['initial_population'])}")
    print(f"Distance matrix shape: {dataset['distance_matrix'].shape}")
    print(f"Time demand matrix shape: {dataset['time_demand_matrix'].shape}")
    
    # Visualize distance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset['distance_matrix'], 
                annot=True, 
                fmt='.0f',
                cmap='YlOrRd')
    plt.title('Distance Matrix Between Cities (km)')
    plt.savefig(f'/BKA_dataset_results/{i}/distance_matrix.png')
    plt.close()
    
    # Visualize demand patterns
    plt.figure(figsize=(12, 6))
    for i in range(params.num_cities):
        for j in range(i+1, params.num_cities):
            demand_pattern = dataset['time_demand_matrix'][i, j, :]
            plt.plot(range(24), demand_pattern, 
                    label=f'Cities {i}-{j}',
                    marker='o')
    
    plt.title('Demand Patterns Between Cities')
    plt.xlabel('Hour of Day')
    plt.ylabel('Passenger Demand')
    plt.grid(True)
    plt.legend()
    plt.savefig('demand_patterns.png')
    plt.close()
    
    return dataset

dataset = test_data_generation()
print("\nTest completed. Check generated visualization files.")