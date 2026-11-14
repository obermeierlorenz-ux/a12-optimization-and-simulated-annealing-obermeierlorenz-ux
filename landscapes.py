import numpy as np
import matplotlib.pyplot as plt
import argparse
from analysis import random_sampling, f_test, function1,function2, contour_plot
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

x_min, y_min, z_min = random_sampling(f_test, rng, N, search_domain1,search_domain2)

print('minimum z test function:', z_min)
print('associated x test function:', x_min)
print('associated y test function:', y_min)

#funktion1

search_domain11 = -2
search_domain12 =  2
x_min1, y_min1, z_min1= random_sampling(function1, rng, N, search_domain11, search_domain12)


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



contour_plot(function1, -2, 2, -2, 2, 'fucntion1', function1_minima)


#funktion2

search_domain21 = -3
search_domain22 =  3
x_min2, y_min2, z_min2 = random_sampling(function2, rng, N, search_domain21, search_domain22)

print()
print('minimum z function 2:', z_min2)
print('associated x function 2:', x_min2)
print('associated y  function 2:', y_min2)


 
#f2 approximate well checks
reference_wells = [
    (-1.85, -1.35),
    (0.15, -1.80),
    (1.65, -0.70),
    (-1.30, 0.75),
    (0.15, 1.75),
    (1.95, 1.45)
]

delta = 0.05

print()
print('Function 2 approximate well checks')

for number, (x_center, y_center) in enumerate(reference_wells, start=1):

    # check the supplied center and four nearby points
    nearby_points = [
        (x_center, y_center),
        (x_center + delta, y_center),
        (x_center - delta, y_center),
        (x_center, y_center + delta),
        (x_center, y_center - delta)
    ]

    nearby_values = []

    for x, y in nearby_points:
        nearby_values.append(function2(x, y))

    lowest_index = np.argmin(nearby_values)

    x_best, y_best = nearby_points[lowest_index]
    z_best = nearby_values[lowest_index]

    print(
        f'Well {number}: '
        f'best nearby point = ({x_best:.3f}, {y_best:.3f}), '
        f'f2 = {z_best:.4f}'
    )

#plot
contour_plot(function2, -3, 3, -3, 3, 'Function 2', reference_wells)