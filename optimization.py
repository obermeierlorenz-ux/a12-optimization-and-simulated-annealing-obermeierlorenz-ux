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
      

    return x, y, z, np.array(x_values), np.array(y_values), np.array(z_values),attempt, stop_reason


def plot_hill_trajectories(
        f,
        x_low,
        x_high,
        y_low,
        y_high,
        runs,
        name
    ):

    x_grid = np.linspace(x_low, x_high, 300)
    y_grid = np.linspace(y_low, y_high, 300)

    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    levels = np.linspace(
        np.min(Z),
        np.percentile(Z, 80),
        30
    )

    plt.figure(figsize=(8, 6))

    contour = plt.contourf(
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

    plt.colorbar(contour, label='Function value')

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for i, run in enumerate(runs):

        color = colors[i]

        plt.plot(
            run['x_path'],
            run['y_path'],
            color=color,
            linewidth=1.2,
            label=f'Run {i + 1}'
        )

        plt.plot(
            run['x0'],
            run['y0'],
            marker='o',
            color=color,
            markersize=7,
            linestyle='none'
        )

        plt.plot(
            run['x_final'],
            run['y_final'],
            marker='X',
            color=color,
            markeredgecolor='black',
            markersize=8,
            linestyle='none'
        )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Hill-Climbing Trajectories: {name}')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_objective_histories(runs, name):

    plt.figure(figsize=(8, 5))

    for i, run in enumerate(runs):

        attempts = np.arange(len(run['z_path']))

        plt.plot(
            attempts,
            run['z_path'],
            label=f'Run {i + 1}'
        )

    plt.xlabel('Attempted step')
    plt.ylabel('Objective value')
    plt.title(f'Objective Value History: {name}')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

#Part 4

#annealing

def simulated_annealing(x0, y0, rng, f,
                        x_low, x_high, y_low, y_high,
                        step_size, T0, alpha, Tmin,
                        N_max = 10000, W=100, eps=1e-4):

    #current state
    x = x0
    y = y0
    z = f(x, y)

    #beste state on the hole run
    best_x = x
    best_y = y
    best_z = z

    #histories
    x_values = [x]
    y_values = [y]
    z_values = [z]
    best_z_values = [best_z]
    temperature_values = [T0]

    stop_reason = 'maximum attempts reached'

    for attempt in range(1, N_max +1):

        temperature = T0 * alpha**(attempt-1)

        angele = rng.uniform(0.0, 2.0*np.pi)

        x_proposed = x + step_size *np.cos(angele)
        y_proposed = y + step_size *np.sin(angele)

        # Only evaluate points inside the domain
        if x_low <= x_proposed <= x_high and y_low <= y_proposed <= y_high:

            z_proposed = f(x_proposed, y_proposed)
            delta_f = z_proposed - z

            # Downhill moves are always accepted
            if delta_f <= 0:
                x = x_proposed
                y = y_proposed
                z = z_proposed

            # Uphill moves are sometimes accepted
            else:
                acceptance_probability = np.exp(-delta_f / temperature)

                if rng.random() < acceptance_probability:
                    x = x_proposed
                    y = y_proposed
                    z = z_proposed

        #update best point if the current point is better
        if z < best_z:
            best_x = x
            best_y = y
            best_z = z

        #store current state after every attempted step
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        best_z_values.append(best_z)

        # cool after every attempt including rejected moves
        final_temperature = T0 * alpha**attempt
        temperature_values.append(final_temperature)

        #stop only if both conditions are fulfilled
        if attempt >= W:

            best_improvement = best_z_values[-W - 1] - best_z_values[-1]

            if final_temperature <= Tmin and best_improvement <= eps:
                stop_reason = 'low temperature and little improvement'
                break

    return (
        x, y, z,  # final current point
        best_x, best_y, best_z,  # best point found
        np.array(x_values),
        np.array(y_values),
        np.array(z_values),
        np.array(best_z_values),
        np.array(temperature_values),
        attempt,
        final_temperature,
        stop_reason
    )

#delta f> 0 accept with probability

#T0 move and cool after every attemped

#track current state and best point encountered