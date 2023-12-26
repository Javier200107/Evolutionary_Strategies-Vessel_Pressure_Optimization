import numpy as np
from deap import base, creator, tools, algorithms

# Definir el problema como un problema de minimización de aptitud
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)



# Modificar initCycle para respetar los límites
# Llamando a func(lower_bound[i], upper_bound[i], size=size y quitando
# el parámetro n de population, igual que en initRepeat se consigue lo mismo
def init_cycle(container, func, n):
    # Definir los límites del problema (bounds)
    upper_bound = [6.1875, 6.1875, 200, 240]
    lower_bound = [0.0625, 0.0625, 10, 10]
    values = [func(lower_bound[i], upper_bound[i]) for i in range(n)]
    return container(values)

# Registrar operadores específicos de tu problema
toolbox = base.Toolbox()
toolbox.register("individual", init_cycle, creator.Individual, np.random.uniform, n=4)
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
    return 0.6224*x1*x3*x4 + 1.7781*x2*x3*x3 + 3.1661*x1*x1*x4 + 19.84*x1*x1*x3

def constraints(individual):
    x = np.array(individual)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    g1 = -x1 + 0.0193*x3
    g2 = -x2 + 0.00954*x3
    g3 = -np.pi*x3*x3*x4 - (4*np.pi/3)*x3*x3*x3 + 1296000
    g4 = x4 - 240
    return [g1, g2, g3, g4]

def feasibility(individual):
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
    return results

# Agregar restricciones a la función de aptitud 
# NO PUEDE SER DIRECTAMENTE EN EL REGISTRO?
#Función de mutación personalizada
def custom_mutate(ind, sigma):
    tau = 1/(np.sqrt(2*np.sqrt(len(ind))))
    tau_prime = 1/(np.sqrt(2*len(ind)))
    r = np.random.normal(0, 1, len(ind))
    child_sigma = sigma * np.exp(tau * r + tau_prime * r)

    r = np.random.normal(0, 1, len(ind))
    child_value = np.copy(ind) + child_sigma * r

    # Aplicar restricciones aquí si es necesario

    return child_value, child_sigma

# Registrar la función de mutación personalizada
toolbox.register("mutate", custom_mutate)
# Agregar la función de aptitud al toolbox
toolbox.register("evaluate", pressure_vessel)

def main():
    pop = toolbox.population(n=10)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 5

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        print('o')

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    print(pop)

if __name__ == "__main__":
    main()
