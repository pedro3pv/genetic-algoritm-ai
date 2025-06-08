import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from collections import Counter
import os

POPULATION_SIZE = 100
MAX_GENERATIONS = 500
MUTATION_RATE = 0.01
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2

NO_IMPROVEMENT_GENERATIONS_THRESHOLD = 50
MIN_COST_IMPROVEMENT_FOR_RESET = 0.001

MIN_VISITABLE_POINTS_TARGET = 31
MAX_VISITABLE_POINTS_TARGET = 59
N_POINTS_TO_SELECT_IF_TOO_MANY = 50

NUM_RUNS_FOR_STATS = 10
ACCEPTABLE_SOLUTION_CRITERION_MET_STAGNATION = "No significant cost improvement"
ACCEPTABLE_SOLUTION_CRITERION_MET_MAX_GENS = "Reached maximum generations"

OUTPUT_IMAGE_DIR = "ga_tsp_results"
SAVE_PLOTS = True
SHOW_PLOTS_AFTER_SAVING = False

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
    if origin_idx == -1:
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

    if n_available_visitable == 0:
        visitable_points_indices_selected = []
    elif MIN_VISITABLE_POINTS_TARGET <= n_available_visitable <= MAX_VISITABLE_POINTS_TARGET:
        visitable_points_indices_selected = all_potential_visitable_indices_full
    elif n_available_visitable > MAX_VISITABLE_POINTS_TARGET:
        num_to_sample = N_POINTS_TO_SELECT_IF_TOO_MANY
        if not (MIN_VISITABLE_POINTS_TARGET <= num_to_sample <= MAX_VISITABLE_POINTS_TARGET):
            num_to_sample = sorted([MIN_VISITABLE_POINTS_TARGET, num_to_sample, MAX_VISITABLE_POINTS_TARGET])[1]

        if num_to_sample > n_available_visitable :
             num_to_sample = n_available_visitable

        visitable_points_indices_selected = random.sample(all_potential_visitable_indices_full, num_to_sample)
    else:
        visitable_points_indices_selected = all_potential_visitable_indices_full
    return points_data_full, origin_idx, visitable_points_indices_selected


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
    if route_distance == 0: return float('inf')
    if route_distance == float('inf'): return 0.0
    return 1.0 / route_distance

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
    if len(parent1) < 2: return parent1[:]
    start_idx, end_idx = sorted(random.sample(range(len(parent1)), 2))
    segment_from_p1 = parent1[start_idx : end_idx + 1]
    child[start_idx : end_idx + 1] = segment_from_p1
    fill_idx = 0
    for gene_p2 in parent2:
        if gene_p2 not in segment_from_p1:
            while child[fill_idx] is not None:
                fill_idx += 1
                if fill_idx >= len(child): break
            if fill_idx < len(child):
                 child[fill_idx] = gene_p2
            else: break
    if None in child:
        elements_in_child = set(c for c in child if c is not None)
        remaining_from_parent1 = [g for g in parent1 if g not in elements_in_child]
        current_fill_idx_fallback = 0
        for i in range(len(child)):
            if child[i] is None:
                if current_fill_idx_fallback < len(remaining_from_parent1):
                    child[i] = remaining_from_parent1[current_fill_idx_fallback]
                    current_fill_idx_fallback += 1
                else: return parent1[:]
    return child

def mutate(individual, mutation_rate=MUTATION_RATE):
    if random.random() < mutation_rate:
        if len(individual) >= 2:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def genetic_algorithm(points_data_full, origin_idx, visitable_points_indices_selected,
                      pop_size, max_gens, mutation_rate_val, tournament_size_val, elitism_count_val,
                      no_improvement_gens_threshold, min_cost_improvement_reset, run_identifier=""):

    if not visitable_points_indices_selected:
        print(f"GA Run {run_identifier}: Cannot run, no points selected to visit.")
        return None, float('inf'), [], "No visitable points", 0

    population = initialize_population(pop_size, visitable_points_indices_selected)
    best_overall_route = None
    best_overall_distance = float('inf')
    history_best_distance = []
    generations_since_last_significant_improvement = 0
    stop_reason = f"Reached maximum generations ({max_gens})."

    print(f"\nStarting GA Run {run_identifier} with Pop: {pop_size}, Gens: {max_gens}, MutRate: {mutation_rate_val*100:.1f}%, TournSize: {tournament_size_val}, Elitism: {elitism_count_val}")
    print(f"Num visitable points: {len(visitable_points_indices_selected)}. Stopping if no improvement > {min_cost_improvement_reset} for {no_improvement_gens_threshold} gens.")

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
            best_overall_distance = current_best_distance_in_pop
            best_overall_route = population[current_best_idx_in_pop][:]
        if not improved_significantly_this_gen and generation > 0 :
            generations_since_last_significant_improvement += 1
        history_best_distance.append(best_overall_distance)
        if (generation + 1) % 100 == 0 or generation == 0 and MAX_GENERATIONS > 50:
             if (generation + 1) % 50 == 0 or generation == 0 :
                print(f"Run {run_identifier}, Gen {generation + 1}/{max_gens} - Best: {best_overall_distance:.2f} (No sig. imp. for {generations_since_last_significant_improvement} gens)")


        if generations_since_last_significant_improvement >= no_improvement_gens_threshold:
            stop_reason = f"Stopped early at gen {generation + 1}: {ACCEPTABLE_SOLUTION_CRITERION_MET_STAGNATION} for {no_improvement_gens_threshold} gens."
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
            if parent1 is None or parent2 is None:
                new_population = population[:]
                break
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=mutation_rate_val)
            new_population.append(child)
        population = new_population
        if not population:
            print(f"Run {run_identifier}: Population became empty. Stopping GA.")
            stop_reason = "Population became empty."
            break
    print(f"\nGA Run {run_identifier} Finished. {stop_reason}")
    print(f"Best overall distance for this run: {best_overall_distance:.4f} after {gen_count} generations.")
    return best_overall_route, best_overall_distance, history_best_distance, stop_reason, gen_count


def plot_route(points_data_full, route_indices, origin_idx, title="TSP Route", save_path=None):
    if not route_indices:
        print("Cannot plot empty route.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    full_path_indices_plot = [origin_idx] + route_indices + [origin_idx]
    path_coords = points_data_full[full_path_indices_plot]
    ax.scatter(points_data_full[:, 0], points_data_full[:, 1], points_data_full[:, 2], c='gray', marker='.', s=5, label='All Loaded Unique Points', alpha=0.2)
    selected_visitable_coords = points_data_full[route_indices]
    ax.scatter(selected_visitable_coords[:, 0], selected_visitable_coords[:, 1], selected_visitable_coords[:, 2], c='blue', marker='o', s=20, label='Selected Visitable Points', alpha=0.8)
    ax.scatter(points_data_full[origin_idx, 0], points_data_full[origin_idx, 1], points_data_full[origin_idx, 2], c='red', s=150, marker='X', label='Origin', depthshade=False, edgecolors='k')
    ax.plot(path_coords[:, 0], path_coords[:, 1], path_coords[:, 2], color='green', linestyle='-', linewidth=1.5, label='Route Path')
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

    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Route plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving route plot to {save_path}: {e}")
    if SHOW_PLOTS_AFTER_SAVING or not save_path :
        plt.show()
    plt.close(fig)


def plot_fitness_history(history_best_distance, title="Cost Convergence", save_path=None):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history_best_distance, marker='.', linestyle='-')
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best Distance (Cost)")
    plt.grid(True)

    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Convergence plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving convergence plot to {save_path}: {e}")
    if SHOW_PLOTS_AFTER_SAVING or not save_path:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    csv_file_path = "CaixeiroGruposGA.csv"
    generations_to_reach_criterion_list = []
    all_run_best_distances = []

    if SAVE_PLOTS and not os.path.exists(OUTPUT_IMAGE_DIR):
        try:
            os.makedirs(OUTPUT_IMAGE_DIR)
            print(f"Created directory: {OUTPUT_IMAGE_DIR}")
        except OSError as e:
            print(f"Error creating directory {OUTPUT_IMAGE_DIR}: {e}. Plots will not be saved.")
            SAVE_PLOTS = False

    for run_num in range(1, NUM_RUNS_FOR_STATS + 1):
        run_id_str = f"run{run_num:02d}"
        print(f"\n=============== STATISTICAL RUN {run_id_str}/{NUM_RUNS_FOR_STATS} ================")
        try:
            current_points_data, current_origin_idx, current_visitable_indices = load_points_numpy(csv_file_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Exiting due to error during data loading: {e}")
            exit()

        if not current_visitable_indices and len(current_points_data) > 1:
            print(f"Run {run_id_str}: TSP cannot run, no points to visit for this selection.")
            continue
        elif len(current_points_data) <= 1:
            print(f"Run {run_id_str}: TSP cannot run, not enough unique points loaded.")
            continue

        best_route_run, best_distance_run, fitness_history_run, stop_reason_text_run, generations_run_count = genetic_algorithm(
            current_points_data, current_origin_idx, current_visitable_indices,
            pop_size=POPULATION_SIZE,
            max_gens=MAX_GENERATIONS,
            mutation_rate_val=MUTATION_RATE,
            tournament_size_val=TOURNAMENT_SIZE,
            elitism_count_val=ELITISM_COUNT,
            no_improvement_gens_threshold=NO_IMPROVEMENT_GENERATIONS_THRESHOLD,
            min_cost_improvement_reset=MIN_COST_IMPROVEMENT_FOR_RESET,
            run_identifier=run_id_str
        )
        all_run_best_distances.append(best_distance_run)

        if ACCEPTABLE_SOLUTION_CRITERION_MET_STAGNATION in stop_reason_text_run or \
           ACCEPTABLE_SOLUTION_CRITERION_MET_MAX_GENS in stop_reason_text_run :
            generations_to_reach_criterion_list.append(generations_run_count)

        if SAVE_PLOTS:
            route_save_filename = None
            convergence_save_filename = None
            if best_route_run:
                route_save_filename = os.path.join(OUTPUT_IMAGE_DIR, f"{run_id_str}_route_cost{best_distance_run:.0f}_pts{len(current_visitable_indices)}.png")
                plot_route(current_points_data, best_route_run, current_origin_idx,
                           title=f"{run_id_str.upper()}: Best Route (Cost: {best_distance_run:.2f}, {len(current_visitable_indices)} pts)",
                           save_path=route_save_filename)
            if fitness_history_run:
                convergence_save_filename = os.path.join(OUTPUT_IMAGE_DIR, f"{run_id_str}_convergence_cost{best_distance_run:.0f}_gens{generations_run_count}.png")
                plot_fitness_history(fitness_history_run,
                                     title=f"{run_id_str.upper()}: Convergence (Final Cost: {best_distance_run:.2f}, Gens: {generations_run_count})",
                                     save_path=convergence_save_filename)
        print(f"================ END OF {run_id_str.upper()} ================\n")


    print("\n\n========== STATISTICAL ANALYSIS SUMMARY ==========")
    if generations_to_reach_criterion_list:
        gen_counts = Counter(generations_to_reach_criterion_list)
        mode_info = gen_counts.most_common(1)
        print(f"List of generations where GA stopped (stagnation/max_gens) across relevant runs: {generations_to_reach_criterion_list}")
        if mode_info:
            mode_generations_value = mode_info[0][0]
            mode_generations_freq = mode_info[0][1]
            print(f"Mode of generations to reach stopping criterion: {mode_generations_value} generations (occurred {mode_generations_freq} times).")
        else: print("No mode could be determined.")
        avg_generations = np.mean(generations_to_reach_criterion_list)
        median_generations = np.median(generations_to_reach_criterion_list)
        print(f"Average generations: {avg_generations:.2f}")
        print(f"Median generations: {median_generations:.2f}")
    else: print("No data collected for generation mode analysis.")

    if all_run_best_distances:
        valid_distances = [d for d in all_run_best_distances if d != float('inf')]
        if valid_distances:
            print(f"\nBest distances found across {NUM_RUNS_FOR_STATS} runs: {[f'{d:.2f}' for d in valid_distances]}")
            print(f"Average best distance: {np.mean(valid_distances):.2f}")
            print(f"Min best distance: {np.min(valid_distances):.2f}")
            print(f"Max best distance: {np.max(valid_distances):.2f}")
            print(f"Std Dev of best distances: {np.std(valid_distances):.2f}")
        else:
            print("\nNo valid distances found across runs.")


    print("\n--- Configuration for Report (based on last run or typical settings) ---")
    print("Methodology:")
    print(f"  - Genetic Algorithm for 3D TSP.")
    print(f"  - Libraries used: NumPy, Matplotlib, CSV, Collections, OS (Python standard library).")
    print(f"  - Population Size (N): {POPULATION_SIZE}")
    print(f"  - Max Generations (per run): {MAX_GENERATIONS}")
    print(f"  - Selection: Tournament (size {TOURNAMENT_SIZE})")
    print(f"  - Crossover: Custom two-point permutation crossover")
    print(f"  - Mutation: Swap mutation (rate {MUTATION_RATE * 100}%)")
    print(f"  - Elitism: {ELITISM_COUNT} best individuals carried over")
    print(f"  - Fitness Function: 1 / Total_Route_Distance (internally, cost/distance directly minimized)")
    print(f"  - Point Selection: Target {MIN_VISITABLE_POINTS_TARGET}-{MAX_VISITABLE_POINTS_TARGET} visitable points.")
    if N_POINTS_TO_SELECT_IF_TOO_MANY and MAX_VISITABLE_POINTS_TARGET < 1000:
        print(f"    If more available, {N_POINTS_TO_SELECT_IF_TOO_MANY} are sampled (or clamped to target range).")
    print(f"  - Stopping criteria (per run):")
    print(f"    - Reaching max generations ({MAX_GENERATIONS}).")
    print(f"    - OR, if no cost improvement > {MIN_COST_IMPROVEMENT_FOR_RESET} for {NO_IMPROVEMENT_GENERATIONS_THRESHOLD} consecutive generations.")
    print("\nResults (Example from statistical runs):")
    if 'mode_generations_value' in locals():
        print(f"  - Mode of generations to stop: {mode_generations_value} (occurred {mode_generations_freq} / {len(generations_to_reach_criterion_list)} relevant runs)")
    if all_run_best_distances and valid_distances:
        print(f"  - Average best distance found over {NUM_RUNS_FOR_STATS} runs: {np.mean(valid_distances):.4f}")
    print("====================================================")