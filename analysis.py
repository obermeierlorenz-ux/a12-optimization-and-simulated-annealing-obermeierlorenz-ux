import numpy as np
import matplotlib.pyplot as plt
import argparse
from optimization import (random_sampling, contour_plot, random_move_hill_climbing, plot_hill_trajectories,
                        plot_objective_histories, simulated_annealing, compare_annealing_schedules,
                        plot_annealing_schedule_trajectories, plot_annealing_schedule_histories,
                        run_annealing_starts, print_annealing_results)
from landscapes import f_test, function1, function2
#test 


#seed
parser = argparse.ArgumentParser()

parser.add_argument(
    '--seed',
    type=int,
    default=44,
    help='Random-number-generator seed'
)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

print('Using seed:', args.seed)

N= 10000

#test function
search_domain1 = -2
search_domain2 =  2

x_min, y_min, z_min, _, _ = random_sampling(f_test, rng, N, search_domain1,search_domain2)

print('minimum z test function:', z_min)
print('associated x test function:', x_min)
print('associated y test function:', y_min)

#funktion1

search_domain11 = -2
search_domain12 =  2
x_min1, y_min1, z_min1, x_samples1, y_samples1= random_sampling(function1, rng, N, search_domain11, search_domain12)


print()
print('minimum z function 1:', z_min1)
print('associated x function 1:', x_min1)
print('associated y  function 1:', y_min1)


#referenve checks
function1_points = {
    'left minimum': (-1.024, 0.524),
    'right minimum': (0.974, 0.474),
    'saddle point': (0.051, 0.0013)
}

print()
print('Function 1 reference checks')

for name, (x, y) in function1_points.items():

    z = function1(x, y)

    print(
        f'{name}: '
        f'x = {x:.4f}, '
        f'y = {y:.4f}, '
        f'f1 = {z:.4f}'
    )

#plot f1
function1_minima = [
    (-1.024, 0.524),
    (0.974, 0.474)
]

contour_plot(function1, -2, 2, -2, 2, 'fucntion1',
              function1_minima, x_samples1, y_samples1, 
              x_min1, y_min1)


#funktion2
search_domain21 = -3
search_domain22 =  3
x_min2, y_min2, z_min2, x_samples2, y_samples2= random_sampling(function2, rng, N, search_domain21, search_domain22)

print()
print('minimum z function 2:', z_min2)
print('associated x function 2:', x_min2)
print('associated y  function 2:', y_min2)


 
#f2 approximate well checks
function2_reference_minima = [
    (1.64544, -0.69885, -5.8069149),
    (0.14972, 1.73740, -5.3369602),
    (-1.84237, -1.34725, -4.6843808),
    (-1.28987, 0.74622, -4.6492501),
    (0.15812, -1.79770, -4.0699216),
    (1.93291, 1.44377, -3.5477975)
]

reference_wells = [
    (x, y) for x, y, z in function2_reference_minima
]

delta = 0.05
print()
print('Function 2 reference-minimum checks')

for number, (x_center, y_center, z_reference) in enumerate(
        function2_reference_minima,
        start=1
    ):

    z_at_reference = function2(x_center, y_center)

    nearby_points = [
        (x_center + delta, y_center),
        (x_center - delta, y_center),
        (x_center, y_center + delta),
        (x_center, y_center - delta)
    ]

    nearby_values = []

    for x, y in nearby_points:
        nearby_values.append(function2(x, y))

    nearby_minimum = np.min(nearby_values)

    print(
        f'Well {number}: '
        f'computed = {z_at_reference:.6f}, '
        f'reference = {z_reference:.6f}, '
        f'lowest nearby value = {nearby_minimum:.6f}'
    )
#plot
contour_plot(function2, -3, 3, -3, 3, 'Function 2', reference_wells,
             x_samples2, y_samples2, x_min2, y_min2)


#Part 3 random move hill climbing

#startingpoints

starts_f1 = [
    (-1.7, -1.2),
    (-1.5, 1.5),
    (-0.2, -1.0),
    (0.0, 1.2),
    (1.4, 1.2),     
    (1.7, -1.5)
]

starts_f2 = [
    (-2.5, -2.5),
    (-2.3, 1.8),
    (-0.5, -1.8),
    (0.5, 1.5),
    (2.4, 1.0),     
    (2.5, -1.5)
]


def compare_step_lengths(
        f,
        x0,
        y0,
        x_low,
        x_high,
        y_low,
        y_high,
        step_sizes,
        seed
    ):


    print()
    print('Step-length comparison')
    print('s | final x | final y | final value | attempts | reason')
    print('-' * 70)

    for step_size in step_sizes:

        # New generator at the beginning of each comparison run
        rng_compare = np.random.default_rng(seed)

        result = random_move_hill_climbing(
            x0,
            y0,
            rng_compare,
            f,
            x_low,
            x_high,
            y_low,
            y_high,
            step_size
        )

        x_final, y_final, z_final, x_path, y_path, z_path, attempts, reason = result

        print(
            f'{step_size:.3f} | '
            f'{x_final:7.3f} | '
            f'{y_final:7.3f} | '
            f'{z_final:11.6f} | '
            f'{attempts:8d} | '
            f'{reason}'
        )

#test ausführen 
step_sizes = [0.02, 0.05, 0.1, 0.2]

# Function 1: required starting point
compare_step_lengths(
    function1,
    1.4,
    1.2,
    -2,
    2,
    -2,
    2,
    step_sizes,
    args.seed
)

# Function 2: required starting point
compare_step_lengths(
    function2,
    2.4,
    1.0,
    -3,
    3,
    -3,
    3,
    step_sizes,
    args.seed
)
# step size
step_size_f1 = 0.05
step_size_f2 = 0.1

def run_hill_climbing_starts(
        f,
        starts,
        x_low,
        x_high,
        y_low,
        y_high,
        step_size,
        rng
    ):

    runs = []

    for x0, y0 in starts:

        result = random_move_hill_climbing(
            x0,
            y0,
            rng,
            f,
            x_low,
            x_high,
            y_low,
            y_high,
            step_size
        )

        x_final, y_final, z_final, x_path, y_path, z_path, attempts, reason = result

        run = {
            'x0': x0,
            'y0': y0,
            'x_final': x_final,
            'y_final': y_final,
            'z_final': z_final,
            'x_path': x_path,
            'y_path': y_path,
            'z_path': z_path,
            'attempts': attempts,
            'reason': reason,
            'step_size': step_size
        }

        runs.append(run)

    return runs

runs_f1 = run_hill_climbing_starts(
    function1,
    starts_f1,
    -2,
    2,
    -2,
    2,
    step_size_f1,
    rng
)

runs_f2 = run_hill_climbing_starts(
    function2,
    starts_f2,
    -3,
    3,
    -3,
    3,
    step_size_f2,
    rng
)

def print_hill_climbing_results(name, runs):

    print()
    print(name)
    print('run | start x | start y | final x | final y | final value | s | attempts | reason')
    print('-' * 100)

    for i, run in enumerate(runs, start=1):

        print(
            f'{i:3d} | '
            f'{run["x0"]:7.3f} | '
            f'{run["y0"]:7.3f} | '
            f'{run["x_final"]:7.3f} | '
            f'{run["y_final"]:7.3f} | '
            f'{run["z_final"]:11.6f} | '
            f'{run["step_size"]:5.3f} | '
            f'{run["attempts"]:8d} | '
            f'{run["reason"]}'
        )

print_hill_climbing_results('Function 1 hill climbing', runs_f1)
print_hill_climbing_results('Function 2 hill climbing', runs_f2)

#plot 

plot_hill_trajectories(
    function1,
    -2,
    2,
    -2,
    2,
    runs_f1,
    'Function 1'
)

plot_hill_trajectories(
    function2,
    -3,
    3,
    -3,
    3,
    runs_f2,
    'Function 2'
)

#plto history

plot_objective_histories(runs_f1, 'Function 1')
plot_objective_histories(runs_f2, 'Function 2')


#part 4 annealing

#test for Function 1
rng_annealing = np.random.default_rng(args.seed)

T0 = 1.0
alpha = 0.999
Tmin = 1e-3

result = simulated_annealing(
    1.0, 0.5, 
    rng_annealing,
    function1,
    -2, 2, -2, 2, 
    step_size_f1,
    T0,
    alpha,
    Tmin,
    N_max=10000,
    W=100,
    eps=1e-4
)

(
    x_final, y_final, z_final,
    best_x, best_y, best_z,
    x_path, y_path, z_path,
    best_z_path, temperature_values,
    attempts, final_temperature,
    reason
) = result

print()
print('Simulated annealing test: Function 1')
print(f'Final point: ({x_final:.5f}, {y_final:.5f})')
print(f'Final value: {z_final:.6f}')
print(f'Best point: ({best_x:.5f}, {best_y:.5f})')
print(f'Best value: {best_z:.6f}')
print(f'Final temperature: {final_temperature:.6f}')
print(f'Attempts: {attempts}')
print(f'Stopping reason: {reason}')



#compare the three tempretures
schedules_f1 = [
    ('low initial temperature', 0.05, 0.999),
    ('sustained exploration', 1.0, 0.999),
    ('rapid cooling', 1.0, 0.99)
]

schedules_f2 = [
    ('low initial temperature', 0.1, 0.999),
    ('sustained exploration', 3.0, 0.999),
    ('rapid cooling', 3.0, 0.99)
]

schedule_results_f1 = compare_annealing_schedules(
    function1,
    1.0, 0.5,           
    -2, 2, -2, 2,
    step_size_f1,
    schedules_f1,
    args.seed,
    'Function 1'
)

schedule_results_f2 = compare_annealing_schedules(
    function2,
    1.95, 1.45,           
    -3, 3, -3, 3,
    step_size_f2,
    schedules_f2,
    args.seed,
    'Function 2'
)



# one representative run from each Function-1 schedule
representative_f1 = [
    schedule_results_f1['low initial temperature'][0],
    schedule_results_f1['sustained exploration'][0],
    schedule_results_f1['rapid cooling'][0]
]

# For Function 2, use a sustained-exploration run which escaped
representative_f2 = [
    schedule_results_f2['low initial temperature'][0],
    schedule_results_f2['sustained exploration'][1],
    schedule_results_f2['rapid cooling'][0]
]

plot_annealing_schedule_trajectories(
    function1,
    -2, 2, -2, 2,
    representative_f1,
    'Function 1'
)

plot_annealing_schedule_histories(
    representative_f1,
    global_minimum=-0.2024,
    initial_value=function1(1.0, 0.5),
    name='Function 1'
)

plot_annealing_schedule_trajectories(
    function2,
    -3, 3, -3, 3,
    representative_f2,
    'Function 2'
)

plot_annealing_schedule_histories(
    representative_f2,
    global_minimum=-5.8069149,
    initial_value=function2(1.95, 1.45),
    name='Function 2'
)




annealing_runs_f1 = run_annealing_starts(
    function1, starts_f1,
    -2, 2, -2, 2,
    step_size_f1,
    T0=1.0, alpha=0.999,
    seed=args.seed
)

annealing_runs_f2 = run_annealing_starts(
    function2, starts_f2,
    -3, 3, -3, 3,
    step_size_f2,
    T0=3.0, alpha=0.999,
    seed=args.seed
)

print_annealing_results(
    'Function 1: selected annealing schedule',
    annealing_runs_f1
)

print_annealing_results(
    'Function 2: selected annealing schedule',
    annealing_runs_f2
)

#plots 

plot_hill_trajectories(
    function1, -2, 2, -2, 2,
    annealing_runs_f1,
    'Function 1 Simulated Annealing'
)

plot_objective_histories(
    annealing_runs_f1,
    'Function 1 Simulated Annealing'
)

plot_hill_trajectories(
    function2, -3, 3, -3, 3,
    annealing_runs_f2,
    'Function 2 Simulated Annealing'
)

plot_objective_histories(
    annealing_runs_f2,
    'Function 2 Simulated Annealing'
)

#comparison 

hill_f1 = random_move_hill_climbing(
    1.0, 0.5, np.random.default_rng(args.seed),
    function1, -2, 2, -2, 2, step_size_f1
)

anneal_f1 = simulated_annealing(
    1.0, 0.5, np.random.default_rng(args.seed),
    function1, -2, 2, -2, 2,
    step_size_f1, 1.0, 0.999, 1e-3
)

hill_f2 = random_move_hill_climbing(
    1.95, 1.45, np.random.default_rng(args.seed),
    function2, -3, 3, -3, 3, step_size_f2
)

anneal_f2 = simulated_annealing(
    1.95, 1.45, np.random.default_rng(args.seed),
    function2, -3, 3, -3, 3,
    step_size_f2, 3.0, 0.999, 1e-3
)

print()
print('Shallow-well comparison')
print('Function 1 hill-climbing value:', hill_f1[2])
print('Function 1 annealing best value:', anneal_f1[5])
print('Function 2 hill-climbing value:', hill_f2[2])
print('Function 2 annealing best value:', anneal_f2[5])