# CI: Optimizing a difficult function with Evolution Strategies

The following code has to be run from root folder in the project. The paths are relative to this location.

## Environment setup + run code
1. Developed on Visual Studio Code using Conda environment + Python 3.8
2. Generate env: `python -m venv env/`
3. Activate environment: 
    - Linux/Mac: `source env/bin/activate`
    - Windows: `env\Scripts\activate.bat`
4. Install requirements: `pip install -r requirements.txt`
5. Run experiments with fixed hyperparameters: `python evolution_strategies_deap.py --mu 1000 --lambda_ 1000 --mutpb 0.8 --ngen 50 --elitism_ratio 0.1 --tournament_size 10 --n_experiments 3`
6. Output will be saved in `output/` folder
7. Results are saved in `output/results.csv` file
8. The progress involving population offspring over generations of the best experiment (minimum fitness value inidividual) is seralized in `output/logbook.pkl` file, which can be loaded using `pickle.load(file)`

# Parameters
- `--mu`: Number of individuals in the population
- `--lambda_`: Number of offspring generated in each generation
- `--mutpb`: Mutation probability
- `--ngen`: Number of generations
- `--elitism_ratio`: Ratio of individuals to be selected for the next generation (elitism)
- `--tournament_size`: Size of the tournament for selection
- `--n_experiments`: Number of experiments to run
 