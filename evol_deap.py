import numpy as np
from deap import base, creator, tools, algorithms

# Definir el problema como un problema de minimización de aptitud
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

# Definir los límites del problema (bounds)
upper_bound = [6.1875, 6.1875, 200, 240]
lower_bound = [0.0625, 0.0625, 10, 10]
bounds = list(zip(lower_bound, upper_bound))

# Registrar operadores específicos de tu problema
toolbox = base.Toolbox()
toolbox.register("individual", tools.initCycle, creator.Individual, (np.random.uniform, ), n=4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Función de aptitud (es necesario cambiar el nombre)
def pressure_vessel(individual):
    # Tu función de aptitud aquí (adaptada a usar el formato DEAP)
    x = np.array(individual)
    x1 = np.round(x[0]/0.0625, 0)*0.0625
    x2 = np.round(x[1]/0.0625, 0)*0.0625
    x3 = x[2]
    x4 = x[3]
    fitness_value = 0.6224*x1*x3*x4 + 1.7781*x2*x3*x3 + 3.1661*x1*x1*x4 + 19.84*x1*x1*x3
    return (fitness_value,)

# Restricciones (cambia el nombre según tu implementación)
def constraints(individual):
    x = np.array(individual)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    g1 = -x1 + 0.0193*x3
    g2 = -x2 + 0.00954*x3
    g3 = -np.pi*x3*x3*x4 - (4*np.pi/3)*x3*x3*x3 + 1296000
    g4 = x4 - 240
    return [g1, g2, g3, g4]

# Restricciones (es necesario cambiar el nombre)
def feasibility(individual):
    return sum(1 for g in constraints(individual) if g > 0),

# Agregar restricciones a la función de aptitud
toolbox.decorate("mate", tools.DeltaPenalty(feasibility, delta=1.0))
toolbox.decorate("mutate", tools.DeltaPenalty(feasibility, delta=1.0))

# Agregar la función de aptitud al toolbox
toolbox.register("evaluate", pressure_vessel)

def main():
    # Configuración de la población y operadores genéticos
    population = toolbox.population(n=1000)
    mu, lambda_ = 50, 100

    # Crear la estadística para el registro
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    stats.register("avg", np.mean)

    # Ejecutar el algoritmo genético
    algorithms.eaMuPlusLambda(population, toolbox, mu=mu, lambda_=lambda_, cxpb=0.7, mutpb=0.2, ngen=100, stats=stats, halloffame=None, verbose=True)

    # Obtener el mejor individuo
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual:", best_ind)
    print("Fitness:", best_ind.fitness.values[0])

    # Visualizar estadísticas
    min_fit = stats.compile(population)["min"]
    avg_fit = stats.compile(population)["avg"]
    gen = range(1, len(min_fit) + 1)

    import matplotlib.pyplot as plt

    plt.plot(gen, min_fit, label="Min Fitness")
    plt.plot(gen, avg_fit, label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
