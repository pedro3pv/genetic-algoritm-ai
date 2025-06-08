import numpy as np
import csv  # For reading CSV files (standard Python library)
import random
import matplotlib.pyplot as plt
from collections import Counter # For mode calculation

# --- Configuration Parameters ---
POPULATION_SIZE = 100
MAX_GENERATIONS = 500  # Max generations for a single GA run
MUTATION_RATE = 0.01
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2

NO_IMPROVEMENT_GENERATIONS_THRESHOLD = 50
MIN_COST_IMPROVEMENT_FOR_RESET = 0.001

# --- Parameters for Point Selection ---
# We want Npontos such that 30 < Npontos < 60. So, min is 31, max is 59.
MIN_VISITABLE_POINTS_TARGET = 31
MAX_VISITABLE_POINTS_TARGET = 59
# If more than MAX_VISITABLE_POINTS_TARGET are available, how many to select:
# Let's aim for a value within the desired range, e.g., 50
N_POINTS_TO_SELECT_IF_TOO_MANY = 50

# --- Parameters for Statistical Analysis of Generations ---
NUM_RUNS_FOR_STATS = 10  # Number of times to run the GA for mode analysis
# Define what constitutes an "acceptable solution" for mode analysis.
# We'll use the GA's own stopping reason: if it stops due to "No significant cost improvement"
# or "Reached maximum generations", we consider the generations_run for that.
ACCEPTABLE_SOLUTION_CRITERION_MET_STAGNATION = "No significant cost improvement"
ACCEPTABLE_SOLUTION_CRITERION_MET_MAX_GENS = "Reached maximum generations"


# --- 1. Data Loading and Preparation (NumPy/CSV version) ---
def load_points_numpy(csv_path="CaixeiroGruposGA.csv"):
    """
    Loads points from the CSV file.
    Selects a subset of visitable points to be between MIN_VISITABLE_POINTS_TARGET
    and MAX_VISITABLE_POINTS_TARGET if possible.
    """
    all_rows_data = []
    try:
        with open(csv_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            for i, row in enumerate(csv_reader):
                if not row: continue
                try:
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    all_rows_data.append(np.array([x, y, z]))
                except ValueError as e:
                    print(f"Warning: Skipping CSV row {i + 1} due to value parsing error: {row} - {e}")
                    continue
                except IndexError as e:
                    print(
                        f"Warning: Skipping CSV row {i + 1} due to insufficient columns: {row} - {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        raise

    if not all_rows_data:
        raise ValueError("No data successfully loaded from CSV.")

    designated_origin_coord = all_rows_data[-1]
    unique_coords_set_tuples = set()
    final_points_list = []

    for p_coord in all_rows_data:
        p_tuple = tuple(p_coord)
        if p_tuple not in unique_coords_set_tuples:
            unique_coords_set_tuples.add(p_tuple)
            final_points_list.append(p_coord)

    points_data_full = np.array(final_points_list)
    origin_idx = -1
    for i, p_coord_unique in enumerate(points_data_full):
        if np.array_equal(p_coord_unique, designated_origin_coord):
            origin_idx = i
            break
    if origin_idx == -1: # Fallback if designated origin (last line) wasn't unique or was problematic
        print(f"Warning: Designated origin {designated_origin_coord} not directly found in unique points. Attempting to find (0,0,0).")
        for i, p_coord_unique in enumerate(points_data_full):
            if np.array_equal(p_coord_unique, np.array([0.0, 0.0, 0.0])):
                origin_idx = i
                designated_origin_coord = points_data_full[i]
                print(f"Using {designated_origin_coord} at index {i} as origin.")
                break
        if origin_idx == -1:
            raise ValueError("Critical: Could not establish a valid origin point. Check CSV data.")


    all_potential_visitable_indices_full = [i for i in range(len(points_data_full)) if i != origin_idx]
    n_available_visitable = len(all_potential_visitable_indices_full)

    print(f"\n--- Point Selection Process ---")
    print(f"Loaded {len(points_data_full)} unique points initially.")
    print(f"Origin: {points_data_full[origin_idx]} at index {origin_idx} in full unique list.")
    print(f"Total potential visitable points (excluding origin): {n_available_visitable}")

    if n_available_visitable == 0:
        visitable_points_indices_selected = []
        print("Warning: No points available to visit after excluding origin.")
    elif MIN_VISITABLE_POINTS_TARGET <= n_available_visitable <= MAX_VISITABLE_POINTS_TARGET:
        visitable_points_indices_selected = all_potential_visitable_indices_full
        print(f"Using all {n_available_visitable} available visitable points (within target range {MIN_VISITABLE_POINTS_TARGET}-{MAX_VISITABLE_POINTS_TARGET}).")
    elif n_available_visitable > MAX_VISITABLE_POINTS_TARGET:
        num_to_sample = N_POINTS_TO_SELECT_IF_TOO_MANY
        # Ensure num_to_sample is also within the desired range if N_POINTS_TO_SELECT_IF_TOO_MANY is outside
        if not (MIN_VISITABLE_POINTS_TARGET <= num_to_sample <= MAX_VISITABLE_POINTS_TARGET):
            print(f"Warning: N_POINTS_TO_SELECT_IF_TOO_MANY ({num_to_sample}) is outside target range. Adjusting selection count.")
            # Default to a value in the middle of the target range or clamp
            num_to_sample = sorted([MIN_VISITABLE_POINTS_TARGET, num_to_sample, MAX_VISITABLE_POINTS_TARGET])[1]

        if num_to_sample > n_available_visitable : # Should not happen if n_available > MAX_TARGET
             num_to_sample = n_available_visitable

        visitable_points_indices_selected = random.sample(all_potential_visitable_indices_full, num_to_sample)
        print(f"Available visitable points ({n_available_visitable}) > {MAX_VISITABLE_POINTS_TARGET}. Randomly selected {len(visitable_points_indices_selected)} points.")
    else: # n_available_visitable < MIN_VISITABLE_POINTS_TARGET (but > 0)
        visitable_points_indices_selected = all_potential_visitable_indices_full
        print(f"Warning: Number of visitable points ({n_available_visitable}) is less than {MIN_VISITABLE_POINTS_TARGET}. Using all available.")

    print(f"Final number of visitable points for GA: {len(visitable_points_indices_selected)}")
    print(f"--------------------------------\n")

    # Note: points_data_full contains ALL unique points.
    # origin_idx is an index into points_data_full.
    # visitable_points_indices_selected contains indices (also into points_data_full) of the points to be visited.
    return points_data_full, origin_idx, visitable_points_indices_selected


# --- 2. Core GA Functions ---
def euclidean_distance_3d(p1, p2):
    return np.linalg.norm(p1 - p2)


def calculate_route_distance(route_indices, points_data, origin_idx):
    if not route_indices: return float('inf')
    total_dist = 0
    # Distance from origin to the first point in the route
    total_dist += euclidean_distance_3d(points_data[origin_idx], points_data[route_indices[0]])
    # Distance between points in the route
    for i in range(len(route_indices) - 1):
        total_dist += euclidean_distance_3d(points_data[route_indices[i]], points_data[route_indices[i + 1]])
    # Distance from the last point in the route back to the origin
    total_dist += euclidean_distance_3d(points_data[route_indices[-1]], points_data[origin_idx])
    return total_dist


def create_individual(visitable_points_indices):
    # visitable_points_indices already contains the actual indices from the main points_data array
    individual = list(visitable_points_indices)
    random.shuffle(individual)
    return individual


def initialize_population(pop_size, visitable_points_indices):
    return [create_individual(visitable_points_indices) for _ in range(pop_size)]


def calculate_fitness(route_distance):
    if route_distance == 0: return float('inf')
    if route_distance == float('inf'): return 0.0
    return 1.0 / route_distance


# --- 3. GA Operators ---
def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    actual_k = min(k, len(population))
    if actual_k == 0: return random.choice(population) if population else None

    tournament_contenders_indices = random.sample(range(len(population)), actual_k)
    best_contender_idx_in_sample = -1
    best_fitness_in_sample = -1.0
    for contender_original_idx in tournament_contenders_indices:
        if fitnesses[contender_original_idx] > best_fitness_in_sample:
            best_fitness_in_sample = fitnesses[contender_original_idx]
            best_contender_idx_in_sample = contender_original_idx
    return population[best_contender_idx_in_sample]


def crossover(parent1, parent2):
    child = [None] * len(parent1)
    if not parent1: return []

    if len(parent1) < 2:
        return parent1[:]

    start_idx, end_idx = sorted(random.sample(range(len(parent1)), 2))
    segment_from_p1 = parent1[start_idx : end_idx + 1]
    child[start_idx : end_idx + 1] = segment_from_p1

    fill_idx = 0
    for gene_p2 in parent2:
        if gene_p2 not in segment_from_p1:
            while child[fill_idx] is not None: # Find next empty spot
                fill_idx += 1
                if fill_idx >= len(child): break # Should not happen if logic is correct
            if fill_idx < len(child):
                 child[fill_idx] = gene_p2
            else: # Safety break, indicates an issue if parents don't have same gene pool
                # print("Crossover error: ran out of space in child.")
                break # This part of parent2 cannot be added.

    # Fallback for any remaining Nones (should not happen in TSP with correct permutation crossover)
    if None in child:
        # print("Warning: None found in child after primary crossover fill. Attempting fallback.")
        elements_in_child = set(c for c in child if c is not None)
        remaining_from_parent1 = [g for g in parent1 if g not in elements_in_child]
        current_fill_idx_fallback = 0
        for i in range(len(child)):
            if child[i] is None:
                if current_fill_idx_fallback < len(remaining_from_parent1):
                    child[i] = remaining_from_parent1[current_fill_idx_fallback]
                    current_fill_idx_fallback += 1
                else: # Should be extremely rare if parents are valid permutations of same set
                    # print("Critical Crossover Error: Could not complete child permutation. Returning parent1.")
                    return parent1[:]
    return child


def mutate(individual, mutation_rate=MUTATION_RATE):
    if random.random() < mutation_rate:
        if len(individual) >= 2:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


# --- 4. Main GA Loop ---
def genetic_algorithm(points_data_full, origin_idx, visitable_points_indices_selected,
                      pop_size, max_gens, mutation_rate_val, tournament_size_val, elitism_count_val,
                      no_improvement_gens_threshold, min_cost_improvement_reset):

    if not visitable_points_indices_selected:
        print("GA cannot run: No points selected to visit.")
        return None, float('inf'), [], "No visitable points", 0

    population = initialize_population(pop_size, visitable_points_indices_selected)
    best_overall_route = None
    best_overall_distance = float('inf')
    history_best_distance = []

    generations_since_last_significant_improvement = 0
    stop_reason = f"Reached maximum generations ({max_gens})."

    print(f"\nStarting GA with Pop: {pop_size}, Gens: {max_gens}, MutRate: {mutation_rate_val*100:.1f}%, TournSize: {tournament_size_val}, Elitism: {elitism_count_val}")
    print(f"Number of points to visit in this run: {len(visitable_points_indices_selected)}")
    print(f"Stopping if no improvement > {min_cost_improvement_reset} for {no_improvement_gens_threshold} gens.")

    gen_count = 0
    for generation in range(max_gens):
        gen_count = generation + 1
        route_distances = [calculate_route_distance(ind, points_data_full, origin_idx) for ind in population]
        fitnesses = [calculate_fitness(dist) for dist in route_distances]

        current_best_idx_in_pop = np.argmin(route_distances)
        current_best_distance_in_pop = route_distances[current_best_idx_in_pop]

        improved_significantly_this_gen = False
        if current_best_distance_in_pop < best_overall_distance:
            if (best_overall_distance - current_best_distance_in_pop) >= min_cost_improvement_reset:
                generations_since_last_significant_improvement = 0
                improved_significantly_this_gen = True
            # else: improvement was too small, don't reset counter (handled below)
            best_overall_distance = current_best_distance_in_pop
            best_overall_route = population[current_best_idx_in_pop][:]

        if not improved_significantly_this_gen and generation > 0 : # only increment if no sig. improvement
            generations_since_last_significant_improvement += 1


        history_best_distance.append(best_overall_distance)

        if (generation + 1) % 50 == 0 or generation == 0:
            print(f"Generation {generation + 1}/{max_gens} - Best Distance: {best_overall_distance:.2f} (No sig. improvement for {generations_since_last_significant_improvement} gens)")

        if generations_since_last_significant_improvement >= no_improvement_gens_threshold:
            stop_reason = f"Stopped early at generation {generation + 1}: {ACCEPTABLE_SOLUTION_CRITERION_MET_STAGNATION} for {no_improvement_gens_threshold} generations."
            # print(f"\n{stop_reason}") # Print moved to end of GA
            break

        new_population = []
        if elitism_count_val > 0 and len(population) > 0:
            actual_elitism_count = min(elitism_count_val, len(population))
            sorted_population_indices = np.argsort(route_distances)
            for i in range(actual_elitism_count):
                new_population.append(population[sorted_population_indices[i]])

        while len(new_population) < pop_size:
            if not population: break
            parent1 = tournament_selection(population, fitnesses, k=tournament_size_val)
            parent2 = tournament_selection(population, fitnesses, k=tournament_size_val)
            if parent1 is None or parent2 is None: # Should not happen if population is not empty
                # print("Warning: Parent selection failed. Skipping generation.")
                new_population = population[:] # Keep current population
                break

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=mutation_rate_val)
            new_population.append(child)

        population = new_population
        if not population:
            print("Population became empty. Stopping GA.")
            stop_reason = "Population became empty."
            break
    # Final print after loop finishes or breaks
    print(f"\nGA Run Finished. {stop_reason}")
    print(f"Best overall distance for this run: {best_overall_distance:.4f} after {gen_count} generations.")
    return best_overall_route, best_overall_distance, history_best_distance, stop_reason, gen_count


# --- 5. Visualization ---
def plot_route(points_data_full, route_indices, origin_idx, title="TSP Route"):
    if not route_indices:
        print("Cannot plot empty route.")
        return

    fig = plt.figure(figsize=(12, 10)) # Increased size
    ax = fig.add_subplot(111, projection='3d')

    # Construct the full path including start and end at origin
    full_path_indices_plot = [origin_idx] + route_indices + [origin_idx]
    path_coords = points_data_full[full_path_indices_plot]

    # Plot all unique points (from which selection was made)
    ax.scatter(points_data_full[:, 0], points_data_full[:, 1], points_data_full[:, 2], c='gray', marker='.', s=5, label='All Loaded Unique Points', alpha=0.2)

    # Plot the selected visitable points
    selected_visitable_coords = points_data_full[route_indices] # route_indices contains the actual points visited
    ax.scatter(selected_visitable_coords[:, 0], selected_visitable_coords[:, 1], selected_visitable_coords[:, 2], c='blue', marker='o', s=20, label='Selected Visitable Points', alpha=0.8)

    # Plot Origin
    ax.scatter(points_data_full[origin_idx, 0], points_data_full[origin_idx, 1], points_data_full[origin_idx, 2], c='red', s=150, marker='X', label='Origin', depthshade=False, edgecolors='k')

    # Plot Route Path
    ax.plot(path_coords[:, 0], path_coords[:, 1], path_coords[:, 2], color='green', linestyle='-', linewidth=1.5, label='Route Path')

    # Annotate start and end of the TSP segment (first and last visited points)
    if route_indices:
        first_visited_actual_idx = route_indices[0]
        last_visited_actual_idx = route_indices[-1]
        ax.text(points_data_full[first_visited_actual_idx, 0], points_data_full[first_visited_actual_idx, 1], points_data_full[first_visited_actual_idx, 2],
                "  Visit Start", color='darkgreen', fontsize=9)
        ax.text(points_data_full[last_visited_actual_idx, 0], points_data_full[last_visited_actual_idx, 1], points_data_full[last_visited_actual_idx, 2],
                "  Visit End", color='darkred', fontsize=9)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(title, fontsize=14)
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

    generations_to_reach_criterion_list = []
    all_run_best_distances = []

    # Perform multiple runs for statistical analysis
    for run_num in range(1, NUM_RUNS_FOR_STATS + 1):
        print(f"\n=============== STATISTICAL RUN {run_num}/{NUM_RUNS_FOR_STATS} ================")
        try:
            # Load and select points FOR EACH RUN if selection involves randomness
            # This ensures that if N_POINTS_TO_SELECT_IF_TOO_MANY is used,
            # different subsets might be chosen for each statistical run,
            # giving a more robust mode analysis for typical problem sizes defined.
            current_points_data, current_origin_idx, current_visitable_indices = load_points_numpy(csv_file_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Exiting due to error during data loading: {e}")
            exit()

        if not current_visitable_indices and len(current_points_data) > 1:
            print("TSP cannot run: No points to visit for this selection.")
            continue # Skip to next stat run
        elif len(current_points_data) <= 1:
            print("TSP cannot run: Not enough unique points loaded.")
            continue # Skip to next stat run

        best_route_run, best_distance_run, fitness_history_run, stop_reason_text_run, generations_run_count = genetic_algorithm(
            current_points_data, current_origin_idx, current_visitable_indices,
            pop_size=POPULATION_SIZE,
            max_gens=MAX_GENERATIONS,
            mutation_rate_val=MUTATION_RATE,
            tournament_size_val=TOURNAMENT_SIZE,
            elitism_count_val=ELITISM_COUNT,
            no_improvement_gens_threshold=NO_IMPROVEMENT_GENERATIONS_THRESHOLD,
            min_cost_improvement_reset=MIN_COST_IMPROVEMENT_FOR_RESET
        )
        all_run_best_distances.append(best_distance_run)

        # Collect generations_run_count if an "acceptable" stop reason occurred
        if ACCEPTABLE_SOLUTION_CRITERION_MET_STAGNATION in stop_reason_text_run or \
           ACCEPTABLE_SOLUTION_CRITERION_MET_MAX_GENS in stop_reason_text_run :
            generations_to_reach_criterion_list.append(generations_run_count)

        # Optionally, plot for the first run or best run, etc.
        if run_num == 1 and best_route_run: # Plot for the first run as an example
             print("\n--- Plotting results for the first run ---")
             plot_route(current_points_data, best_route_run, current_origin_idx,
                        title=f"Run 1: Best Route (Cost: {best_distance_run:.2f}, {len(current_visitable_indices)} pts)")
             plot_fitness_history(fitness_history_run,
                                  title=f"Run 1: Convergence (Final Cost: {best_distance_run:.2f}, Gens: {generations_run_count})")
        print(f"================ END OF RUN {run_num} ================\n")


    # --- After all statistical runs, perform analysis ---
    print("\n\n========== STATISTICAL ANALYSIS SUMMARY ==========")
    if generations_to_reach_criterion_list:
        gen_counts = Counter(generations_to_reach_criterion_list)
        mode_info = gen_counts.most_common(1) # Returns list of [(value, count)]

        print(f"List of generations where GA stopped (stagnation/max_gens) across {NUM_RUNS_FOR_STATS} runs: {generations_to_reach_criterion_list}")
        if mode_info:
            mode_generations_value = mode_info[0][0]
            mode_generations_freq = mode_info[0][1]
            print(f"Mode of generations to reach stopping criterion: {mode_generations_value} generations (occurred {mode_generations_freq} times).")
        else:
            print("No mode could be determined (e.g., all generation counts were unique or list empty).")

        avg_generations = np.mean(generations_to_reach_criterion_list)
        median_generations = np.median(generations_to_reach_criterion_list)
        print(f"Average generations: {avg_generations:.2f}")
        print(f"Median generations: {median_generations:.2f}")
    else:
        print("No data collected for generation mode analysis (e.g., GA never met criteria).")

    if all_run_best_distances:
        print(f"\nBest distances found across {NUM_RUNS_FOR_STATS} runs: {[f'{d:.2f}' for d in all_run_best_distances]}")
        print(f"Average best distance: {np.mean(all_run_best_distances):.2f}")
        print(f"Min best distance: {np.min(all_run_best_distances):.2f}")
        print(f"Max best distance: {np.max(all_run_best_distances):.2f}")
        print(f"Std Dev of best distances: {np.std(all_run_best_distances):.2f}")


    print("\n--- Configuration for Report (based on last run or typical settings) ---")
    print("Methodology:")
    print(f"  - Genetic Algorithm for 3D TSP.")
    print(f"  - Libraries used: NumPy, Matplotlib, CSV (Python standard library), Collections (for mode).")
    print(f"  - Population Size (N): {POPULATION_SIZE}")
    print(f"  - Max Generations (per run): {MAX_GENERATIONS}")
    print(f"  - Selection: Tournament (size {TOURNAMENT_SIZE})")
    print(f"  - Crossover: Custom two-point permutation crossover")
    print(f"  - Mutation: Swap mutation (rate {MUTATION_RATE * 100}%)")
    print(f"  - Elitism: {ELITISM_COUNT} best individuals carried over")
    print(f"  - Fitness Function: 1 / Total_Route_Distance (internally, cost/distance directly minimized)")
    print(f"  - Point Selection: Target {MIN_VISITABLE_POINTS_TARGET}-{MAX_VISITABLE_POINTS_TARGET} visitable points.")
    if N_POINTS_TO_SELECT_IF_TOO_MANY and MAX_VISITABLE_POINTS_TARGET < 1000: # Avoid printing if it's effectively "all"
        print(f"    If more available, {N_POINTS_TO_SELECT_IF_TOO_MANY} are sampled (or clamped to target range).")
    print(f"  - Stopping criteria (per run):")
    print(f"    - Reaching max generations ({MAX_GENERATIONS}).")
    print(f"    - OR, if no cost improvement > {MIN_COST_IMPROVEMENT_FOR_RESET} for {NO_IMPROVEMENT_GENERATIONS_THRESHOLD} consecutive generations.")
    print("\nResults (Example from statistical runs):")
    if 'mode_generations_value' in locals(): # Check if variable exists
        print(f"  - Mode of generations to stop: {mode_generations_value} (occurred {mode_generations_freq} / {NUM_RUNS_FOR_STATS} times)")
    if all_run_best_distances:
        print(f"  - Average best distance found over {NUM_RUNS_FOR_STATS} runs: {np.mean(all_run_best_distances):.4f}")
    print("====================================================")