import numpy as np
import matplotlib.pyplot as plt
import argparse
from optimization import random_sampling, contour_plot, random_move_hill_climbing
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
x0 = 1.4
y0 = 1.2
length = 0.02
M = 10

x_final, y_final, z_final, x_path, y_path, z_path, attempts, reason = random_move_hill_climbing(
    1.4,
    1.2,
    rng,
    function1,
    -2,
    2,
    -2,
    2,
    step_size=0.02
)
print ()
print('Random-move hill climbing method')
print('final x:', x_final)
print('final y:', y_final)
print('final value:', z_final)
print('attempted steps:', attempts)
print('stopping reason:', reason)