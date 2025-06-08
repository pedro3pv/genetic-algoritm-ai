import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

# --- Configuration Parameters ---
# These can be tuned based on the problem and experimentation
POPULATION_SIZE = 100        # N: Number of individuals in the population
MAX_GENERATIONS = 500      # Maximum number of generations
MUTATION_RATE = 0.01       # Probability of mutation (1%)
TOURNAMENT_SIZE = 3         # Size of the tournament for selection
ELITISM_COUNT = 2           # Ne: Number of best individuals to carry to next generation

# --- 1. Data Loading and Preparation ---
def load_points(csv_path="CaixeiroGruposGA.csv"):
    """Loads points from the CSV file."""
    df = pd.read_csv(csv_path, header=None, names=['x', 'y', 'z', 'group'])
    
    # Identify the origin (last point as per problem description)
    # The CSV might have multiple (0,0,0) if points are near origin, but one is designated.
    # We will treat the explicit (0,0,0,0) line as the origin.
    # If there are other (0,0,0) points with different group IDs, they are treated as visitable.
    
    # Store all points as numpy arrays for easier calculation
    all_points_coords = [np.array([row['x'], row['y'], row['z']]) for _, row in df.iterrows()]
    
    # The last point in the CSV is the designated origin
    origin_coord = all_points_coords[-1]
    
    # Visitable points are all points EXCEPT the designated origin point itself.
    # We need to be careful if (0,0,0) appears elsewhere but is not the designated origin.
    # For simplicity, we'll use all unique points, then identify the origin index.
    
    unique_coords_list = []
    unique_coords_set = set()
    for i, p_coord in enumerate(all_points_coords):
        p_tuple = tuple(p_coord)
        if p_tuple not in unique_coords_set:
            unique_coords_set.add(p_tuple)
            unique_coords_list.append(p_coord)
            
    points_data = np.array(unique_coords_list)
    
    # Find the index of the origin within our unique points_data
    origin_idx = -1
    for i, p_coord in enumerate(points_data):
        if np.array_equal(p_coord, origin_coord):
            origin_idx = i
            break
    
    if origin_idx == -1:
        # This case should ideally not happen if origin is in the file
        # Add it if it was somehow filtered out by uniqueness (e.g. if origin was not last row)
        points_data = np.vstack([points_data, origin_coord])
        origin_idx = len(points_data) - 1
        print("Warning: Origin had to be re-added. Check CSV format.")

    visitable_points_indices = [i for i in range(len(points_data)) if i != origin_idx]
    
    print(f"Loaded {len(points_data)} unique points.")
    print(f"Origin coordinates: {points_data[origin_idx]} at index {origin_idx}")
    print(f"Number of other visitable points: {len(visitable_points_indices)}")
    
    # The constraint "30 < Npontos < 60" might refer to the number of visitable_points_indices.
    # If you need to enforce this, you'd filter `visitable_points_indices` here.
    # For now, we use all unique non-origin points.
    if not (30 < len(visitable_points_indices) < 60):
        print(f"Warning: Number of visitable points ({len(visitable_points_indices)}) is outside the 30-60 range.")
        print("The GA will proceed with all loaded unique points.")
        # If strict adherence is needed, one might sample or filter groups here.
        # Example (if you wanted to sample):
        # if len(visitable_points_indices) > 59:
        #     visitable_points_indices = random.sample(visitable_points_indices, 50) # Sample 50 points
        #     print(f"Sampled down to {len(visitable_points_indices)} visitable points.")


    return points_data, origin_idx, visitable_points_indices

# --- 2. Core GA Functions ---
def euclidean_distance_3d(p1, p2):
    """Calculates 3D Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

def calculate_route_distance(route_indices, points_data, origin_idx):
    """
    Calculates the total distance of a route.
    A route_indices is a permutation of visitable_points_indices.
    The full path is origin -> route_indices[0] -> ... -> route_indices[-1] -> origin.
    """
    if not route_indices: # Empty route
        return 0 # Or a very large number if empty routes are invalid

    total_dist = 0
    
    # Distance from origin to the first point in the route
    total_dist += euclidean_distance_3d(points_data[origin_idx], points_data[route_indices[0]])
    
    # Distance between points in the route
    for i in range(len(route_indices) - 1):
        total_dist += euclidean_distance_3d(points_data[route_indices[i]], points_data[route_indices[i+1]])
        
    # Distance from the last point in the route back to origin
    total_dist += euclidean_distance_3d(points_data[route_indices[-1]], points_data[origin_idx])
    
    return total_dist

def create_individual(visitable_points_indices):
    """Creates a random individual (a permutation of visitable points)."""
    individual = list(visitable_points_indices) # Make a mutable copy
    random.shuffle(individual)
    return individual

def initialize_population(pop_size, visitable_points_indices):
    """Initializes the population with random individuals."""
    return [create_individual(visitable_points_indices) for _ in range(pop_size)]

def calculate_fitness(route_distance):
    """Calculates fitness (inverse of distance). Higher is better."""
    if route_distance == 0:
        return float('inf') # Avoid division by zero, though unlikely for TSP
    return 1.0 / route_distance

# --- 3. GA Operators ---

# Selection: Tournament Selection
def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    """Selects an individual using tournament selection."""
    tournament_contenders_indices = random.sample(range(len(population)), k)
    best_contender_idx = -1
    best_fitness = -1.0
    
    for contender_idx in tournament_contenders_indices:
        if fitnesses[contender_idx] > best_fitness:
            best_fitness = fitnesses[contender_idx]
            best_contender_idx = contender_idx
            
    return population[best_contender_idx]

# Recombination: Custom Two-Point Crossover for Permutations
def crossover(parent1, parent2):
    """
    Performs a custom two-point crossover suitable for permutation-based problems.
    1. A random segment from parent1 is copied to the child.
    2. The remaining genes are filled from parent2 in order, excluding duplicates.
    """
    child = [None] * len(parent1)
    
    # 1. Select a random segment from parent1
    start_idx, end_idx = sorted(random.sample(range(len(parent1)), 2))
    
    # Copy the segment from parent1 to the child
    segment_from_p1 = parent1[start_idx : end_idx + 1]
    child[start_idx : end_idx + 1] = segment_from_p1
    
    # 2. Fill remaining genes from parent2
    current_child_idx = 0
    for gene_p2 in parent2:
        if gene_p2 not in segment_from_p1: # Avoid duplicates
            while child[current_child_idx] is not None: # Find next empty spot
                current_child_idx += 1
            child[current_child_idx] = gene_p2
            
    return child

# Mutation: Swap Mutation
def mutate(individual, mutation_rate=MUTATION_RATE):
    """Performs swap mutation on an individual."""
    if random.random() < mutation_rate:
        if len(individual) >= 2: # Need at least two genes to swap
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# --- 4. Main GA Loop ---
def genetic_algorithm(points_data, origin_idx, visitable_points_indices,
                      pop_size, max_gens, mutation_rate_val, tournament_size_val, elitism_count_val):
    """Runs the Genetic Algorithm."""
    
    population = initialize_population(pop_size, visitable_points_indices)
    best_overall_route = None
    best_overall_distance = float('inf')
    
    history_best_distance = []

    print(f"\nStarting GA with Pop: {pop_size}, Gens: {max_gens}, MutRate: {mutation_rate_val}, TournSize: {tournament_size_val}, Elitism: {elitism_count_val}")

    for generation in range(max_gens):
        # Calculate fitness for each individual
        route_distances = [calculate_route_distance(ind, points_data, origin_idx) for ind in population]
        fitnesses = [calculate_fitness(dist) for dist in route_distances]
        
        # Find best in current generation
        current_best_idx = np.argmax(fitnesses)
        current_best_distance = route_distances[current_best_idx]
        
        if current_best_distance < best_overall_distance:
            best_overall_distance = current_best_distance
            best_overall_route = population[current_best_idx]
        
        history_best_distance.append(best_overall_distance)

        if (generation + 1) % 50 == 0: # Print progress
            print(f"Generation {generation + 1}/{max_gens} - Best Distance: {best_overall_distance:.2f}")

        # Create new population
        new_population = []
        
        # Elitism: Carry over the best individuals
        if elitism_count_val > 0:
            sorted_population_indices = np.argsort(fitnesses)[::-1] # Sort by fitness descending
            for i in range(elitism_count_val):
                new_population.append(population[sorted_population_indices[i]])
        
        # Fill the rest of the new population
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, k=tournament_size_val)
            parent2 = tournament_selection(population, fitnesses, k=tournament_size_val)
            
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=mutation_rate_val)
            
            new_population.append(child)
            
        population = new_population
        
    print(f"\nGA Finished. Best overall distance: {best_overall_distance:.2f}")
    return best_overall_route, best_overall_distance, history_best_distance


# --- 5. Visualization (Optional but helpful) ---
def plot_route(points_data, route_indices, origin_idx, title="TSP Route"):
    """Plots the 3D route."""
    if not route_indices:
        print("Cannot plot empty route.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Full path including origin
    full_path_indices = [origin_idx] + route_indices + [origin_idx]
    path_coords = points_data[full_path_indices]

    # Plot points
    ax.scatter(points_data[:, 0], points_data[:, 1], points_data[:, 2], c='blue', marker='o', label='All Points')
    ax.scatter(points_data[origin_idx, 0], points_data[origin_idx, 1], points_data[origin_idx, 2], c='red', s=100, marker='X', label='Origin')
    
    # Plot visited points in order (excluding origin for this highlight)
    visited_coords = points_data[route_indices]
    ax.scatter(visited_coords[:,0], visited_coords[:,1], visited_coords[:,2], c='green', s=50, marker='o', label='Visited in Route')


    # Plot path
    ax.plot(path_coords[:, 0], path_coords[:, 1], path_coords[:, 2], color='green', linestyle='-', linewidth=1.5, label='Route')

    # Annotate start/end points of the actual TSP part (first and last visitable point)
    ax.text(points_data[route_indices[0],0], points_data[route_indices[0],1], points_data[route_indices[0],2], "Start", color='black')
    ax.text(points_data[route_indices[-1],0], points_data[route_indices[-1],1], points_data[route_indices[-1],2], "End", color='black')


    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_fitness_history(history_best_distance, title="Fitness Convergence"):
    plt.figure(figsize=(10, 6))
    plt.plot(history_best_distance, marker='.')
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best Distance (Cost)")
    plt.grid(True)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load data
    points_data, origin_idx, visitable_points_indices = load_points()

    if not visitable_points_indices:
        print("Error: No visitable points found. Check data or loading logic.")
    else:
        # 2. Run GA
        # You can experiment with different parameters here
        best_route, best_distance, fitness_history = genetic_algorithm(
            points_data, origin_idx, visitable_points_indices,
            pop_size=POPULATION_SIZE, 
            max_gens=MAX_GENERATIONS, 
            mutation_rate_val=MUTATION_RATE, 
            tournament_size_val=TOURNAMENT_SIZE, 
            elitism_count_val=ELITISM_COUNT
        )

        print("\n--- Results ---")
        print(f"Best route (indices of visitable points): {best_route}")
        # To get actual coordinates for the report:
        # best_route_coords = [points_data[i] for i in best_route]
        # print(f"Best route coordinates (excluding origin start/end): {best_route_coords}")
        print(f"Total distance of best route: {best_distance:.4f}")
        
        # 3. Analysis/Mode of generations (as requested)
        # "Faça uma análise se de qual é a moda de gerações para obter uma solução aceitável."
        # This typically means: run the GA multiple times and see around which generation
        # an "acceptable" solution (e.g., within X% of the best known, or when improvements stagnate)
        # is usually found. For a single run, we can see when major improvements stop.
        # For a proper mode, you'd run this many times.
        if fitness_history:
            improvements = np.diff(fitness_history)
            stagnation_point = np.where(improvements >= -0.001 * best_distance)[0] # e.g. improvement less than 0.1%
            if len(stagnation_point) > 0:
                 # Find the last point of significant improvement
                last_sig_improvement_gen = 0
                for i in range(len(fitness_history) - 2, 0, -1):
                    if fitness_history[i] - fitness_history[i+1] > 0.001 * best_distance: # More than 0.1% improvement
                        last_sig_improvement_gen = i
                        break
                print(f"Major improvements seemed to slow down around generation: {last_sig_improvement_gen +1}")


        # 4. Plotting (optional, but good for reports)
        if best_route:
            plot_route(points_data, best_route, origin_idx, title=f"Best TSP Route Found (Distance: {best_distance:.2f})")
        if fitness_history:
            plot_fitness_history(fitness_history, title=f"Convergence (Final Distance: {best_distance:.2f})")
            
        print("\n--- For your Report ---")
        print("Methodology:")
        print(f"  - Genetic Algorithm for 3D TSP.")
        print(f"  - Population Size (N): {POPULATION_SIZE}")
        print(f"  - Max Generations: {MAX_GENERATIONS}")
        print(f"  - Selection: Tournament (size {TOURNAMENT_SIZE})")
        print(f"  - Crossover: Custom two-point permutation crossover")
        print(f"  - Mutation: Swap mutation (rate {MUTATION_RATE*100}%)")
        print(f"  - Elitism: {ELITISM_COUNT} best individuals carried over")
        print(f"  - Fitness Function: 1 / Total_Route_Distance")
        print(f"  - Number of unique points (excluding origin for permutation): {len(visitable_points_indices)}")
        print("\nResults:")
        print(f"  - Best distance found: {best_distance:.4f}")
        # print(f"  - Best route (sequence of visitable point indices): {best_route}")
        # Add details from fitness_history or multiple runs for "mode of generations"