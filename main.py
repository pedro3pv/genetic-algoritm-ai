import numpy as np
import csv  # For reading CSV files (standard Python library)
import random
import matplotlib.pyplot as plt

# --- Configuration Parameters ---
POPULATION_SIZE = 100
MAX_GENERATIONS = 500
MUTATION_RATE = 0.01
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2

NO_IMPROVEMENT_GENERATIONS_THRESHOLD = 50
MIN_COST_IMPROVEMENT_FOR_RESET = 0.001


# --- 1. Data Loading and Preparation (NumPy/CSV version) ---
def load_points_numpy(csv_path="CaixeiroGruposGA.csv"):
    """Loads points from the CSV file using only built-in Python, csv module, and NumPy."""
    all_rows_data = []  # To store np.array([x,y,z]) from each row
    try:
        with open(csv_path, 'r', newline='') as file:  # Added newline='' for csv module best practice
            csv_reader = csv.reader(file)
            for i, row in enumerate(csv_reader):
                if not row: continue  # Skip empty lines
                try:
                    # Expecting x, y, z, group_id. We only need x, y, z for coordinates.
                    # The problem states the CSV is (x, y, z, group_id)
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    # group = int(row[3]) # group is not used for TSP path calculation here
                    all_rows_data.append(np.array([x, y, z]))
                except ValueError as e:
                    print(f"Warning: Skipping CSV row {i + 1} due to value parsing error: {row} - {e}")
                    continue
                except IndexError as e:
                    print(
                        f"Warning: Skipping CSV row {i + 1} due to insufficient columns (expected at least 3 for x,y,z): {row} - {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        raise  # Re-raise to be caught by main

    if not all_rows_data:
        raise ValueError("No data successfully loaded from CSV. Check file content and format.")

    # The last point in the CSV is the designated origin, as per problem spec "O último ponto (0,0,0,0) é a origem."
    designated_origin_coord = all_rows_data[-1]

    # Create a list of unique points.
    # The order of points_data will be based on first appearance in the CSV.
    unique_coords_set_tuples = set()  # Use a set of tuples for efficient uniqueness checking
    final_points_list = []

    for p_coord in all_rows_data:
        p_tuple = tuple(p_coord)  # Tuples are hashable for sets
        if p_tuple not in unique_coords_set_tuples:
            unique_coords_set_tuples.add(p_tuple)
            final_points_list.append(p_coord)  # Store the original np.array

    points_data = np.array(final_points_list)

    # Find the index of the designated_origin_coord in the unique points_data.
    # This point *must* exist in points_data because it came from all_rows_data.
    origin_idx = -1
    for i, p_coord_unique in enumerate(points_data):
        if np.array_equal(p_coord_unique, designated_origin_coord):
            origin_idx = i
            break

    if origin_idx == -1:
        # This should not happen if designated_origin_coord was in all_rows_data
        # and then processed into final_points_list.
        # Could happen if designated_origin_coord had NaN values from an empty last line parsed as float.
        # However, the problem implies the origin is (0,0,0).
        # Fallback: try to find a (0,0,0) point if designated one is problematic.
        print(
            f"Warning: Designated origin {designated_origin_coord} not directly found in unique points. Attempting to find (0,0,0).")
        for i, p_coord_unique in enumerate(points_data):
            if np.array_equal(p_coord_unique, np.array([0.0, 0.0, 0.0])):
                origin_idx = i
                designated_origin_coord = points_data[i]  # Update to the found one
                print(f"Using {designated_origin_coord} at index {i} as origin.")
                break
        if origin_idx == -1:
            raise ValueError(
                "Critical: Could not establish a valid origin point. Check CSV data, especially the last line.")

    visitable_points_indices = [i for i in range(len(points_data)) if i != origin_idx]

    print(f"Loaded {len(points_data)} unique points using NumPy/CSV.")
    print(f"Designated Origin (target from last CSV line): {designated_origin_coord}")
    print(f"Effective Origin in points_data: {points_data[origin_idx]} at index {origin_idx}")
    print(f"Number of other visitable points: {len(visitable_points_indices)}")

    # Constraint check from problem description
    if len(visitable_points_indices) > 0:  # Only warn if there are points to visit
        if not (30 < len(visitable_points_indices) < 60):
            print(
                f"Warning: Number of visitable points ({len(visitable_points_indices)}) is outside the 30-60 range suggested in problem description.")
            print("The GA will proceed with the loaded number of visitable points.")
    elif len(points_data) > 1:  # More than one unique point, but all are the origin
        print(f"Warning: All {len(points_data)} unique points seem to be the origin. No points to visit.")
    elif len(points_data) <= 1:
        print(f"Warning: Only {len(points_data)} unique point(s) loaded. TSP is trivial or not possible.")

    return points_data, origin_idx, visitable_points_indices


# --- 2. Core GA Functions ---
def euclidean_distance_3d(p1, p2):
    return np.linalg.norm(p1 - p2)


def calculate_route_distance(route_indices, points_data, origin_idx):
    if not route_indices: return float('inf')
    total_dist = 0
    total_dist += euclidean_distance_3d(points_data[origin_idx], points_data[route_indices[0]])
    for i in range(len(route_indices) - 1):
        total_dist += euclidean_distance_3d(points_data[route_indices[i]], points_data[route_indices[i + 1]])
    total_dist += euclidean_distance_3d(points_data[route_indices[-1]], points_data[origin_idx])
    return total_dist


def create_individual(visitable_points_indices):
    individual = list(visitable_points_indices)
    random.shuffle(individual)
    return individual


def initialize_population(pop_size, visitable_points_indices):
    return [create_individual(visitable_points_indices) for _ in range(pop_size)]


def calculate_fitness(route_distance):
    if route_distance == 0: return float('inf')  # Should not happen for TSP if dist > 0
    if route_distance == float('inf'): return 0.0  # For empty/invalid routes
    return 1.0 / route_distance


# --- 3. GA Operators ---
def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    # Ensure k is not larger than population size, can happen with small test populations
    actual_k = min(k, len(population))
    if actual_k == 0: return random.choice(population)  # Should not happen if pop > 0

    tournament_contenders_indices = random.sample(range(len(population)), actual_k)
    best_contender_idx = -1
    best_fitness = -1.0
    for contender_idx in tournament_contenders_indices:
        if fitnesses[contender_idx] > best_fitness:
            best_fitness = fitnesses[contender_idx]
            best_contender_idx = contender_idx
    return population[best_contender_idx]


def crossover(parent1, parent2):
    child = [None] * len(parent1)
    if not parent1: return []  # Handle empty parent case, though unlikely

    # Ensure start_idx and end_idx are valid for list length
    # len(parent1) must be at least 2 for sample(..., 2)
    if len(parent1) < 2:
        return parent1[:]  # Or handle as an error/special case

    start_idx, end_idx = sorted(random.sample(range(len(parent1)), 2))

    segment_from_p1 = parent1[start_idx: end_idx + 1]
    child[start_idx: end_idx + 1] = segment_from_p1

    current_child_idx = 0
    for gene_p2 in parent2:
        if gene_p2 not in segment_from_p1:
            # Find next empty spot for genes from parent2
            # This loop needs to handle the child list filling up
            filled = False
            for i in range(len(child)):  # Iterate through available slots in child
                if child[i] is None:
                    child[i] = gene_p2
                    filled = True
                    break
            if not filled and gene_p2 not in child:  # Should not happen if logic is correct
                # This case means child is full but gene_p2 wasn't placed.
                # This implies an issue if len(parent1) == len(parent2) and all elements are unique.
                # For TSP, this should be fine.
                pass

    # Ensure child is fully formed and a valid permutation
    if None in child:
        # This can happen if parent2 doesn't have all missing elements
        # (e.g. if parents are of different types/lengths, not for this TSP)
        # Fill remaining Nones with elements from parent1 not already in child
        # This is a fallback, ideally the above logic should cover permutation crossover
        elements_in_child = set(c for c in child if c is not None)
        remaining_from_parent1 = [g for g in parent1 if g not in elements_in_child]

        current_fill_idx = 0
        for i in range(len(child)):
            if child[i] is None:
                if current_fill_idx < len(remaining_from_parent1):
                    child[i] = remaining_from_parent1[current_fill_idx]
                    current_fill_idx += 1
                else:
                    # This is a problem - not enough unique elements to fill.
                    # For TSP, all individuals should have same set of unique genes.
                    # This indicates a deeper issue if reached.
                    # print("Error: Crossover could not form a complete child permutation.")
                    # Fallback to one of the parents or a copy to avoid None values
                    return parent1[:] if random.random() < 0.5 else parent2[:]

    return child


def mutate(individual, mutation_rate=MUTATION_RATE):
    if random.random() < mutation_rate:
        if len(individual) >= 2:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


# --- 4. Main GA Loop ---
def genetic_algorithm(points_data, origin_idx, visitable_points_indices,
                      pop_size, max_gens, mutation_rate_val, tournament_size_val, elitism_count_val,
                      no_improvement_gens_threshold, min_cost_improvement_reset):
    if not visitable_points_indices:
        print("GA cannot run: No points to visit (visitable_points_indices is empty).")
        return None, float('inf'), [], "No visitable points", 0

    population = initialize_population(pop_size, visitable_points_indices)
    best_overall_route = None
    best_overall_distance = float('inf')
    history_best_distance = []

    generations_since_last_significant_improvement = 0
    stop_reason = f"Reached maximum generations ({max_gens})."

    print(
        f"\nStarting GA with Pop: {pop_size}, Gens: {max_gens}, MutRate: {mutation_rate_val}, TournSize: {tournament_size_val}, Elitism: {elitism_count_val}")
    print(
        f"Stopping if no improvement for {no_improvement_gens_threshold} gens (min cost improvement: {min_cost_improvement_reset}).")

    gen_count = 0
    for generation in range(max_gens):
        gen_count = generation + 1
        route_distances = [calculate_route_distance(ind, points_data, origin_idx) for ind in population]
        fitnesses = [calculate_fitness(dist) for dist in route_distances]  # Fitness for selection

        current_best_idx_in_pop = np.argmin(route_distances)  # Min distance is best
        current_best_distance_in_pop = route_distances[current_best_idx_in_pop]

        if current_best_distance_in_pop < best_overall_distance:
            if best_overall_distance - current_best_distance_in_pop >= min_cost_improvement_reset:
                generations_since_last_significant_improvement = 0
            else:  # Improvement was too small
                generations_since_last_significant_improvement += 1
            best_overall_distance = current_best_distance_in_pop
            best_overall_route = population[current_best_idx_in_pop][:]  # Store a copy
        else:
            generations_since_last_significant_improvement += 1

        history_best_distance.append(best_overall_distance)

        if (generation + 1) % 50 == 0 or generation == 0:
            print(
                f"Generation {generation + 1}/{max_gens} - Best Distance: {best_overall_distance:.2f} (No sig. improvement for {generations_since_last_significant_improvement} gens)")

        if generations_since_last_significant_improvement >= no_improvement_gens_threshold:
            stop_reason = f"Stopped early at generation {generation + 1}: No significant cost improvement for {no_improvement_gens_threshold} generations."
            print(f"\n{stop_reason}")
            break

        new_population = []
        if elitism_count_val > 0 and len(population) > 0:
            # Ensure elitism_count is not more than population size
            actual_elitism_count = min(elitism_count_val, len(population))
            sorted_population_indices = np.argsort(route_distances)  # Lower distance is better
            for i in range(actual_elitism_count):
                new_population.append(population[sorted_population_indices[i]])

        while len(new_population) < pop_size:
            if not population: break  # Should not happen if pop_size > 0
            parent1 = tournament_selection(population, fitnesses, k=tournament_size_val)
            parent2 = tournament_selection(population, fitnesses, k=tournament_size_val)

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=mutation_rate_val)

            new_population.append(child)

        population = new_population
        if not population:  # Safety break if population somehow becomes empty
            print("Population became empty. Stopping GA.")
            stop_reason = "Population became empty."
            break

    print(f"\nGA Finished. {stop_reason}")
    print(f"Best overall distance: {best_overall_distance:.2f} after {gen_count} generations.")
    return best_overall_route, best_overall_distance, history_best_distance, stop_reason, gen_count


# --- 5. Visualization ---
def plot_route(points_data, route_indices, origin_idx, title="TSP Route"):
    if not route_indices:
        print("Cannot plot empty route.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    full_path_indices = [origin_idx] + route_indices + [origin_idx]
    path_coords = points_data[full_path_indices]

    ax.scatter(points_data[:, 0], points_data[:, 1], points_data[:, 2], c='blue', marker='o', s=10, label='All Points',
               alpha=0.3)
    ax.scatter(points_data[origin_idx, 0], points_data[origin_idx, 1], points_data[origin_idx, 2], c='red', s=100,
               marker='X', label='Origin', depthshade=False)

    visited_coords_in_route = points_data[route_indices]
    ax.scatter(visited_coords_in_route[:, 0], visited_coords_in_route[:, 1], visited_coords_in_route[:, 2], c='lime',
               s=30, marker='o', label='Visited (in order)', depthshade=False, edgecolors='k', linewidths=0.5)

    ax.plot(path_coords[:, 0], path_coords[:, 1], path_coords[:, 2], color='green', linestyle='-', linewidth=1.5,
            label='Route Path')

    if route_indices:  # Add text for start and end of the TSP part of the route
        first_visited_idx = route_indices[0]
        last_visited_idx = route_indices[-1]
        ax.text(points_data[first_visited_idx, 0], points_data[first_visited_idx, 1], points_data[first_visited_idx, 2],
                " Start Visit", color='darkgreen', fontsize=9)
        ax.text(points_data[last_visited_idx, 0], points_data[last_visited_idx, 1], points_data[last_visited_idx, 2],
                " End Visit", color='darkred', fontsize=9)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_fitness_history(history_best_distance, title="Cost Convergence"):
    plt.figure(figsize=(10, 6))
    plt.plot(history_best_distance, marker='.', linestyle='-')
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best Distance (Cost)")
    plt.grid(True)
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    csv_file_path = "CaixeiroGruposGA.csv"
    try:
        points_data, origin_idx, visitable_points_indices = load_points_numpy(csv_file_path)
    except (FileNotFoundError, ValueError) as e:  # Catch errors from loading
        print(f"Exiting due to error: {e}")
        exit()

    # Check if TSP is runnable
    if not visitable_points_indices and len(points_data) > 1:
        print(
            "TSP cannot run: No points to visit (all unique points might be the origin, or only one unique point was effectively non-origin).")
    elif len(points_data) <= 1:
        print("TSP cannot run: Not enough unique points loaded (need at least one origin and one point to visit).")
    else:
        best_route, best_distance, fitness_history, stop_reason_text, generations_run = genetic_algorithm(
            points_data, origin_idx, visitable_points_indices,
            pop_size=POPULATION_SIZE,
            max_gens=MAX_GENERATIONS,
            mutation_rate_val=MUTATION_RATE,
            tournament_size_val=TOURNAMENT_SIZE,
            elitism_count_val=ELITISM_COUNT,
            no_improvement_gens_threshold=NO_IMPROVEMENT_GENERATIONS_THRESHOLD,
            min_cost_improvement_reset=MIN_COST_IMPROVEMENT_FOR_RESET
        )

        print("\n--- Results ---")
        print(f"Algorithm Stop Reason: {stop_reason_text}")
        print(f"Total generations run: {generations_run}")
        # print(f"Best route (indices of visitable points): {best_route if best_route else 'N/A'}") # Can be long
        print(f"Total distance of best route: {best_distance:.4f}")

        if best_route:
            plot_route(points_data, best_route, origin_idx, title=f"Best TSP Route (Cost: {best_distance:.2f})")
        if fitness_history:
            plot_fitness_history(fitness_history,
                                 title=f"Convergence (Final Cost: {best_distance:.2f}, Gens: {generations_run})")

        print("\n--- For your Report ---")
        print("Methodology:")
        print(f"  - Genetic Algorithm for 3D TSP.")
        print(f"  - Libraries used: NumPy, Matplotlib, CSV (Python standard library).")
        print(f"  - Population Size (N): {POPULATION_SIZE}")
        print(f"  - Max Generations: {MAX_GENERATIONS}")
        print(f"  - Selection: Tournament (size {TOURNAMENT_SIZE})")
        print(f"  - Crossover: Custom two-point permutation crossover")
        print(f"  - Mutation: Swap mutation (rate {MUTATION_RATE * 100}%)")
        print(f"  - Elitism: {ELITISM_COUNT} best individuals carried over")
        print(
            f"  - Fitness Function: 1 / Total_Route_Distance (internally, cost/distance directly minimized for 'best')")
        print(f"  - Number of unique points in dataset: {len(points_data)}")
        print(f"  - Number of visitable points (excluding origin): {len(visitable_points_indices)}")
        print(f"  - Stopping criteria:")
        print(f"    - Reaching max generations ({MAX_GENERATIONS}).")
        print(
            f"    - OR, if no cost improvement greater than {MIN_COST_IMPROVEMENT_FOR_RESET} for {NO_IMPROVEMENT_GENERATIONS_THRESHOLD} consecutive generations.")
        print("\nResults:")
        print(f"  - Algorithm stopped because: {stop_reason_text}")
        print(f"  - Actual generations executed: {generations_run}")
        print(f"  - Best distance found: {best_distance:.4f}")