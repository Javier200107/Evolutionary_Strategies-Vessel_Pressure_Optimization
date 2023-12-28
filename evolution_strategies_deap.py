import numpy as np, operator
from deap import base, creator, tools, algorithms
from functools import partial

# Function to initialize an individual with strategy
def init_individual(icls, scls):
    n = 4
    upper_bound_values = [6.1875, 6.1875, 100, 200]
    lower_bound_values = [0.0625, 0.0625, 10, 10]
    upper_bound_strategy = [1, 1, 5, 10]
    lower_bound_strategy = [0.1, 0.1, 1, 1]
    ind = icls(np.random.uniform(lower_bound_values[i], upper_bound_values[i]) for i in range(n))
    ind.strategy = scls(np.random.uniform(lower_bound_strategy[i], upper_bound_strategy[i]) for i in range(n))
    return ind

# Fitness function (name change is necessary)
def pressure_vessel(individual):
    x = np.array(individual)
    x1 = np.round(x[0]/0.0625, 0)*0.0625
    x2 = np.round(x[1]/0.0625, 0)*0.0625
    x3 = x[2]
    x4 = x[3]
    return (0.6224*x1*x3*x4 + 1.7781*x2*x3*x3 + 3.1661*x1*x1*x4 + 19.84*x1*x1*x3,)

# Constraint functions
def constraints(individual):
    x = np.array(individual)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    g1 = -x1 + 0.0193*x3
    g2 = -x2 + 0.00954*x3
    g3 = -np.pi*x3*x3*x4 - (4*np.pi/3)*x3*x3*x3 + 1296000
    g4 = x4 - 240
    g5 = - pressure_vessel(individual)[0]
    g6 = 0.0625 - x1
    g7 = 0.0625 - x2
    g8 = x1 - 6.1875
    g9 = x2 - 6.1875
    g10 = x3 - 200
    g11 = x4 - 200
    g12 = 10 - x3
    g13 = 10 - x4

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13]

# Feasibility function
def feasibility(individual):
    return not any(g > 0 for g in constraints(individual))

# Distance function for penalizing infeasible individuals
def distance(individual):
    if any(g > 0 for g in constraints(individual)):
        return 1e8

# Filter infeasible individuals
def filter_infeasible(population, function, penal=1e8):
    return function([ind[0] for ind in population if ind[0] < penal])

# Check strategy values
def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy[i]:
                        child.strategy[i] = minstrategy[i]
            return children
        return wrappper
    return decorator
    
# Plotting statistics
def plot_and_save_statistics(logbook, df_results, output_dir="output", params=None):
    
    import pickle, os, datetime, matplotlib.pyplot as plt

    # Create a folder with the current date and time
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_folder = os.path.join(output_dir, timestamp)
    os.makedirs(output_folder, exist_ok=True)

    # Save the logbook to a binary file
    with open(f'{output_folder}/logbook.pkl', 'wb') as file:
        pickle.dump(logbook, file)

    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    size_avgs = logbook.select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    # for tl in ax1.get_yticklabels():
    #     tl.set_color("b")  
    # Use log scale for y-axis
    # ax1.set_yscale('log')

    ax2 = ax1
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Fitness")
    # Use log scale for y-axis
    # ax2.set_yscale('log')

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    if params is not None:
        str_params = " - ".join([f"{key}: {value}" for key, value in params.items()])
        plt.title("Fitness over Generations\n"+str_params)
    else:
        plt.title("Fitness over Generations\n ")

    # Save the figure in the output folder
    output_path = os.path.join(output_folder, 'min_vs_avg.png')
    fig.savefig(output_path, bbox_inches='tight')

    # Boxplot of the best and mean fitness values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot([df_results["Best Fitness"], df_results["Mean Fitness"], df_results["Median Fitness"]], 
               labels=["Best Fitness", "Mean Fitness", "Median Fitness"])
    ax.set_title("Boxplot of Best, Mean and Median Fitness Values")
    ax.set_ylabel("Fitness")

    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the figure in the output folder
    output_path = os.path.join(output_folder, 'boxplot.png')
    plt.savefig(output_path, bbox_inches='tight')

    # boxplot for experiment time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(df_results["Experiment Time"], labels=["Experiment Time"])
    ax.set_title("Boxplot of Experiment Time")
    ax.set_ylabel("Time (s)")

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure in the output folder
    output_path = os.path.join(output_folder, 'boxplot_time.png')
    plt.savefig(output_path, bbox_inches='tight')

    # Save the results to a CSV file
    df_results.to_csv(f'{output_folder}/results.csv', index=False)

    # Write the parameters to a text file
    if params is not None:
        with open(f'{output_folder}/params.txt', 'w') as file:
            for key, value in params.items():
                file.write(f"{key}: {value}\n")

if __name__ == "__main__":
    import pandas as pd
    # Evolutionary Strategies (ES) Algorithm parameters
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Set parameters for the experiment.")

    # Add arguments
    parser.add_argument("--mu", type=int, default=1000, help="Value for MU (default: 1000)")
    parser.add_argument("--lambda_", type=int, default=1000, dest="lambda_", help="Value for LAMBDA_ (default: 1000)")
    parser.add_argument("--mutpb", type=float, default=0.8, help="Value for MUTPB (default: 0.8)")
    parser.add_argument("--ngen", type=int, default=100, help="Value for NGEN (default: 100)")
    parser.add_argument("--elitism_ratio", type=float, default=0.1, help="Value for ELITISM_RATIO (default: 0.1)")
    parser.add_argument("--tournament_size", type=int, default=10, help="Value for TOURNAMENT_SIZE (default: 10)")
    parser.add_argument("--n_experiments", type=int, default=3, help="Number of experiments to run (default: 3)")

    # Parse the arguments
    args = parser.parse_args()

    # Assign variables
    MU, LAMBDA_ = args.mu, args.lambda_
    MUTPB, NGEN = args.mutpb, args.ngen
    ELITISM_RATIO = args.elitism_ratio
    TOURNAMENT_SIZE = args.tournament_size
    N_EXPERIMENTS = args.n_experiments

    # Calculate ELITISM
    ELITISM = int(LAMBDA_ * ELITISM_RATIO)

    # Define the problem as a minimization fitness problem
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin, strategy=None)
    creator.create("Strategy", np.ndarray)

    # Register specific operators for the problem
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, creator.Individual, creator.Strategy)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.3)
    toolbox.decorate("mutate", checkStrategy([0.1, 0.1, 1, 1]))
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Register the evaluation function and apply penalty for infeasibility
    toolbox.register("evaluate", pressure_vessel)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasibility, delta=0, distance=distance))

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", partial(filter_infeasible, function=np.mean))
    stats.register("min", partial(filter_infeasible, function=np.min))
    stats.register("max", partial(filter_infeasible, function=np.max))
    stats.register("std", partial(filter_infeasible, function=np.std))
    stats.register("median", partial(filter_infeasible, function=np.median))
    stats.register("feasibles", partial(filter_infeasible, function=len))
    stats.register("unique_ind", partial(filter_infeasible, function=lambda x: len(np.unique(x))))

    # List to store the results of each experiment
    all_experiments_results = []
    ls_logbooks = []
    for experiment in range(N_EXPERIMENTS):
        # Initial population
        pop = toolbox.population(n=MU)

        print(f"\n\nExperiment {experiment + 1} of {N_EXPERIMENTS}...\n")
        # Evolutionary process and logging for each experiment
        # get time
        import time
        start_time = time.time()
        pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA_,
                                                cxpb=0, mutpb=MUTPB, ngen=NGEN, stats=stats,
                                                halloffame=tools.HallOfFame(ELITISM, lambda x, y: (x == y).all()), 
                                                verbose=False)
        experiment_time = time.time() - start_time

        # Analysis of results for each experiment
        best_ind = tools.selBest(pop, 1)[0]
        arr_fitness = [ind.fitness.values[0] for ind in pop]
        arr_fitness = [x for x in arr_fitness if x < 100000000]
        mean_fitness = np.mean(arr_fitness)
        median_fitness = np.median(arr_fitness)
        worst_fitness = np.max(arr_fitness)
        std_dev = np.std(arr_fitness)
        num_unique_individuals = len(np.unique(arr_fitness))
        num_feasible = len(arr_fitness)
        num_infeasible = len(pop) - len(arr_fitness)

        ls_logbooks.append(logbook)

        if num_feasible == 0:
            print("SKIP INFEASIBLE EXPERIMENT")
            continue

        # Store the results of this experiment
        experiment_results = {
            "Experiment": experiment + 1,
            "Best Fitness": best_ind.fitness.values[0],
            "Mean Fitness": mean_fitness,
            "Median Fitness": median_fitness,
            "Worst Fitness": worst_fitness,
            "Standard Deviation": std_dev,
            "Experiment Time": experiment_time,
            "Number of Unique Individuals": num_unique_individuals,
            "Number of Feasible Individuals": num_feasible,
            "Number of Infeasible Individuals": num_infeasible,
            "Best Individual": best_ind,
            "Constraints": constraints(best_ind),
            # "logbook": logbook,
        }
        all_experiments_results.append(experiment_results)

        # Display results for each experiment
        print("Best individual after", NGEN, "generations:")
        print("Fitness value (Min):", best_ind.fitness.values[0])
        print("Fitness value (Mean):", mean_fitness)
        print("Fitness value (Median):", median_fitness)
        print("Fitness value (Worst):", worst_fitness)
        print("Number of feasible individuals:", num_feasible)
        print("Number of infeasible individuals:", num_infeasible)
        print("Standard deviation:", std_dev)
        print("Number of unique individuals:", num_unique_individuals)
        print("Individual variables:", best_ind)
        print("Experiment time:", experiment_time)
        print("Constraints:", constraints(best_ind))

        # Plot statistics for each experiment
        # plot_and_save_statistics(logbook, params={"MU": MU, "LAMBDA": LAMBDA_, "MUTPB": MUTPB, "NGEN": NGEN, "ELITISM_RATIO": ELITISM_RATIO})

    # Convert the list of results to a DataFrame
    df_results = pd.DataFrame(all_experiments_results).sort_values(by="Best Fitness", ascending=True).reset_index(drop=True)
    best_result_id = df_results.loc[0, "Experiment"] - 1
    d_params = {"MU": MU, "LAMBDA": LAMBDA_, "MUTPB": MUTPB, "NGEN": NGEN, "ELITISM_RATIO": ELITISM_RATIO,
                "TOURNAMENT_SIZE": TOURNAMENT_SIZE}

    ls_measures_cols = ['Best Fitness', 'Mean Fitness', 'Median Fitness', 'Worst Fitness', 'Standard Deviation', 'Number of Unique Individuals', 'Number of Feasible Individuals', 'Number of Infeasible Individuals']
    print(df_results[ls_measures_cols].agg(['mean', 'std', 'min', 'max']).T)

    print("\n\nBest individual after", NGEN, "generations:")
    print(df_results.loc[0, :"Number of Infeasible Individuals"])
    print("Best individual variables:", df_results.loc[0, "Best Individual"])

    print("\n\nParameters:")
    print(d_params)

    best_logbook = ls_logbooks[best_result_id]
    plot_and_save_statistics(best_logbook, df_results, params=d_params)
    # Save 