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
toolbox.register("mate", tools.cxBlend, alpha=0.5)
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
toolbox.decorate("mate", tools.DeltaPenalty(feasibility, delta=1.0))
toolbox.decorate("mutate", tools.DeltaPenalty(feasibility, delta=1.0))

