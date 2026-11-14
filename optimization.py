import numpy as np  
import matplotlib.pyplot as plt
from landscapes import function1, function2, f_test


def random_sampling(f, rng, N, sd1, sd2):

    z_values = []
    x_values = []
    y_values = []
    #funktion

    #random werte im bereich der funktion bestimmen x und y
    for i in range(N):
        x = rng.uniform(sd1, sd2)
        y = rng.uniform(sd1, sd2)

        z = f(x, y)

        z_values.append(z)
        x_values.append(x)
        y_values.append(y)

        min_index = np.argmin(z_values)

        z_min = z_values[min_index]
        x_min = x_values[min_index]
        y_min = y_values[min_index]



    return   x_min, y_min, z_min, np.array(x_values), np.array(y_values)


    #dazugehörigen funktionswert bestimmen 

#plot function

def contour_plot(
        f,
        x_low,
        x_high,
        y_low,
        y_high,
        name,
        minima,
        x_samples,
        y_samples,
        best_x,
        best_y
    ):

    x = np.linspace(x_low, x_high, 300)
    y = np.linspace(y_low, y_high, 300)

    X, Y = np.meshgrid(x, y)

    Z = f(X, Y)

    # Focus the contour levels on low-value regions
    levels = np.linspace(
        np.min(Z),
        np.percentile(Z, 80),
        30
    )

    plt.figure()

    contours = plt.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap='viridis'
    )

    plt.contour(
        X,
        Y,
        Z,
        levels=levels,
        colors='black',
        linewidths=0.4
    )

    plt.colorbar(contours, label='Function value')

    number_to_plot = min(len(x_samples), 2000)

    sample_indices = np.linspace(
        0,
        len(x_samples) - 1,
        number_to_plot,
        dtype=int
    )

    plt.scatter(
        x_samples[sample_indices],
        y_samples[sample_indices],
        s=5,
        color='white',
        alpha=0.35,
        label=f'{number_to_plot} sampled points'
    )

    # Mark all known or approximate minima
    for i, (x_minimum, y_minimum) in enumerate(minima):

        if i == 0:
            label = 'Reference minima'
        else:
            label = None

        plt.plot(
            x_minimum,
            y_minimum,
            marker='*',
            color='red',
            markeredgecolor='black',
            markersize=11,
            linestyle='none',
            label=label
        )

    # mark the best point found by random sampling
    plt.plot(
        best_x,
        best_y,
        marker='X',
        color='yellow',
        markeredgecolor='black',
        markersize=9,
        linestyle='none',
        label='Best sampled point'
    )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Contour Plot: {name}')
    plt.legend()
    plt.grid()

    plt.show()

def random_move_hill_climbing(x0, y0,rng, f, x_low, x_high, y_low, y_high,
                              step_size, N_max=10000, W=100, eps= 1e-4):

    x = x0
    y = y0
    z = f(x, y)

    z_values = [z]
    x_values = [x]
    y_values = [y]

    stop_reason = 'maximum attempts reached'

    for attempt in range(1, N_max +1
                          ):
        angle = rng.uniform(0.0, 2*np.pi)

        x_proposed = x + step_size * np.cos(angle)
        y_proposed = y + step_size * np.sin(angle)

        #only accept wehn lower
        if (x_low <= x_proposed <= x_high and y_low <= y_proposed <= y_high):
            z_proposed = f(x_proposed, y_proposed)

            if z_proposed < z:
                x = x_proposed
                y = y_proposed
                z = z_proposed

        z_values.append(z)
        x_values.append(x)
        y_values.append(y)

        #check total improvement during the previpus W attempts
        if attempt >= W:

            decrease = z_values[-W - 1] - z_values[-1]

            if decrease <= eps:
                stop_reason = 'little improvment'
                break
      

    return x, z, y, np.array(x_values), np.array(y_values), np.array(z_values),attempt, stop_reason

        