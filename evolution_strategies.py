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
    return [g1, g2, g3, g4]

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
def plot_and_save_statistics(logbook, output_dir="output", params=None):
    
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
    for tl in ax1.get_yticklabels():
        tl.set_color("b")  
    # Use log scale for y-axis
    ax1.set_yscale('log')

    ax2 = ax1
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Fitness")
    # Use log scale for y-axis
    ax2.set_yscale('log')

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    if params is not None:
        str_params = " - ".join([f"{key}: {value}" for key, value in params.items()])
        plt.title("Fitness over Generations\n"+str_params)
    else:
        plt.title("Fitness over Generations\n ")
    plt.show()

    # Save the figure in the output folder
    output_path = os.path.join(output_folder, 'min_vs_avg.png')
    fig.savefig(output_path, bbox_inches='tight')

if __name__ == "__main__":

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
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Register the evaluation function and apply penalty for infeasibility
    toolbox.register("evaluate", pressure_vessel)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasibility, delta=7.0, distance=distance))

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", partial(filter_infeasible, function=np.mean))
    stats.register("min", partial(filter_infeasible, function=np.min))
    stats.register("max", partial(filter_infeasible, function=np.max))
    stats.register("std", partial(filter_infeasible, function=np.std))
    stats.register("median", partial(filter_infeasible, function=np.median))
    stats.register("feasibles", partial(filter_infeasible, function=len))
    stats.register("unique_ind", partial(filter_infeasible, function=lambda x: len(np.unique(x))))

    # Evolutionary Strategies (ES) Algorithm parameters
    MU, LAMBDA_ = 1000, 1000
    MUTPB, NGEN = 0.8, 500
    ELITISM_RATIO = 0.1
    ELITISM = int(LAMBDA_*ELITISM_RATIO)

    # Initial population
    pop = toolbox.population(n=MU)

    # Evolutionary process and logging
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA_,
                                            cxpb=0, mutpb=MUTPB, ngen=NGEN, stats=stats,
                                            halloffame=tools.HallOfFame(ELITISM, lambda x, y: (x == y).all()), verbose=True)
    # Display results
    print("Best individual after", NGEN, "generations:")
    best_ind = tools.selBest(pop, 1)[0]
    print("Fitness value (Min):", best_ind.fitness.values[0])
    arr_fitness = [ind.fitness.values[0] for ind in pop]
    arr_fitness = [x for x in arr_fitness if x < 100000000]
    print("Fitness value (Mean):", np.mean(arr_fitness))
    print("Fitness value (Median):", np.median(arr_fitness))
    print("Fitness value (Worst):", np.max(arr_fitness))
    print("Number of feasible individuals:", len(arr_fitness))
    print("Number of infeasible individuals:", len(pop) - len(arr_fitness))
    print("Standard deviation:", np.std(arr_fitness))
    print("Number of unique individuals:", len(np.unique(arr_fitness)))
    print("Individual variables:", best_ind)
    print("Constraints:", constraints(best_ind))

    # Plot statistics
    plot_and_save_statistics(logbook, params={"MU": MU, "LAMBDA": LAMBDA_, "MUTPB": MUTPB, "NGEN": NGEN, "ELITISM_RATIO": ELITISM_RATIO})
