import numpy as np
import matplotlib.pyplot as plt
import argparse
from analysis import random_sampling, f_test, function1
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

z_min, y_min, x_min = random_sampling(f_test, rng, N, search_domain1,search_domain2)

print('minimum z test function:', z_min)
print('associated x test function:', x_min)
print('associated y test function:', y_min)

#funktion1

search_domain11 = -2
search_domain12 =  2
z_min1, y_min1, x_min1 = random_sampling(function1, rng, N, search_domain11, search_domain12)

print()
print('minimum z function 1:', z_min1)
print('associated x function 1:', x_min1)
print('associated y  function 1:', y_min1)

#function 2

#funktion2

search_domain21 = -3
search_domain22 =  3
z_min2, y_min2, x_min2 = random_sampling(function1, rng, N, search_domain21, search_domain22)

print()
print('minimum z function 2:', z_min2)
print('associated x function 2:', x_min2)
print('associated y  function 2:', y_min2)