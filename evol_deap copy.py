import numpy as np
from deap import base, creator, tools, algorithms

# Definir el problema como un problema de minimización de aptitud
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", np.ndarray)

# Modificar initCycle para respetar los límites
# Llamando a func(lower_bound[i], upper_bound[i], size=size y quitando
# el parámetro n de population, igual que en initRepeat se consigue lo mismo
"""def init_cycle(container, func, n):
    # Definir los límites del problema (bounds)
    upper_bound = [6.1875, 6.1875, 200, 240]
    lower_bound = [0.0625, 0.0625, 10, 10]
    values = [func(lower_bound[i], upper_bound[i]) for i in range(n)]
    return container(values)"""
# Function to initialize an individual with strategy
def init_individual(icls, scls):
    n = 4
    upper_bound_values = [6.1875, 6.1875, 200, 240]
    lower_bound_values = [0.0625, 0.0625, 10, 10]
    upper_bound_strategy = [1, 1, 5, 10]
    lower_bound_strategy = [0.1, 0.1, 1, 1]
    ind = icls(np.random.uniform(lower_bound_values[i], upper_bound_values[i]) for i in range(n))
    ind.strategy = scls(np.random.uniform(lower_bound_strategy[i], upper_bound_strategy[i]) for i in range(n))
    return ind
# Registrar operadores específicos de tu problema
toolbox = base.Toolbox()
toolbox.register("individual", init_individual, creator.Individual, creator.Strategy)
# Population should take into account the bounds
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Función de aptitud (es necesario cambiar el nombre)
def pressure_vessel(individual):
    x = np.array(individual)
    x1 = np.round(x[0]/0.0625, 0)*0.0625
    x2 = np.round(x[1]/0.0625, 0)*0.0625
    x3 = x[2]
    x4 = x[3]
    return (0.6224*x1*x3*x4 + 1.7781*x2*x3*x3 + 3.1661*x1*x1*x4 + 19.84*x1*x1*x3,)

def constraints(individual):
    x = np.array(individual)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    g1 = -x1 + 0.0193*x3
    g2 = -x2 + 0.00954*x3
    g3 = -np.pi*x3*x3*x4 - (4*np.pi/3)*x3*x3*x3 + 1296000
    g4 = x4 - 240
    return [g1, g2, g3, g4]

def feasibility(individual):
    return not any(g > 0 for g in constraints(individual))

def distance(individual):
    ls_constraints = constraints(individual)
    if any(g > 0 for g in constraints(individual)):
        return 100000000

"""def feasibility(individual):
    #return sum(1 for g in constraints(individual) if g > 0)
    for i, g in enumerate(constraints(individual)):
        results = []
        broken_constraints = 0
        if g > 0:
            results.append(i)
            broken_constraints += 1
        else:
            results.append(0)
    results.append(broken_constraints)
    return results"""

# Agregar restricciones a la función de aptitud 
# NO PUEDE SER DIRECTAMENTE EN EL REGISTRO?
#Función de mutación personalizada
"""def custom_mutate(ind, sigma):
    tau = 1/(np.sqrt(2*np.sqrt(len(ind))))
    tau_prime = 1/(np.sqrt(2*len(ind)))
    r = np.random.normal(0, 1, len(ind))
    child_sigma = sigma * np.exp(tau * r + tau_prime * r)

    r = np.random.normal(0, 1, len(ind))
    child_value = np.copy(ind) + child_sigma * r

    # Aplicar restricciones aquí si es necesario

    return child_value, child_sigma"""

"""# Registrar la función de mutación personalizada
toolbox.register("mutate", custom_mutate)"""


# Agregar la función de aptitud al toolbox
toolbox.register("evaluate", pressure_vessel)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasibility, delta=7.0, distance=distance))

# Estadísticas
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

# Algoritmo de Estrategias Evolutivas (ES)
MU, LAMBDA_ = 10, 100
CXPB, MUTPB, NGEN = 0.6, 0.3, 50

# Población inicial
pop = toolbox.population(n=MU)
# Estadísticas y registro de población
pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA_,
                                          cxpb=0, mutpb=MUTPB, ngen=NGEN, stats=stats,
                                          halloffame=None, verbose=True)

# Muestra de resultados
print("Mejor individuo después de", NGEN, "generaciones:")
best_ind = tools.selBest(pop, 1)[0]
print("Valor de aptitud:", best_ind.fitness.values[0])
print("Variables del individuo:", best_ind)
print("Constraints:", constraints(best_ind))