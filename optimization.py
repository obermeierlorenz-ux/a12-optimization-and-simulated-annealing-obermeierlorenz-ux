import numpy as np  
import matplotlib.pyplot as plt
from landscapes import function1, function2, f_test

#validation
def validate_search_parameters(
        x_low, x_high, y_low, y_high,
        step_size=None, N_max=None, W=None, eps=None,
        T0=None, alpha=None, Tmin=None
    ):

    if x_low >= x_high or y_low >= y_high:
        raise ValueError('Domain limits must satisfy low < high.')

    if step_size is not None and step_size <= 0:
        raise ValueError('step_size must be positive.')

    if N_max is not None and N_max <= 0:
        raise ValueError('N_max must be positive.')

    if W is not None and (W <= 0 or W > N_max):
        raise ValueError('W must be positive and no larger than N_max.')

    if eps is not None and eps < 0:
        raise ValueError('eps must be non-negative.')

    if T0 is not None and T0 <= 0:
        raise ValueError('T0 must be positive.')

    if Tmin is not None and (Tmin <= 0 or Tmin >= T0):
        raise ValueError('Tmin must satisfy 0 < Tmin < T0.')

    if alpha is not None and not 0 < alpha < 1:
        raise ValueError('alpha must satisfy 0 < alpha < 1.')
    

def random_sampling(f, rng, N, sd1, sd2):

    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError('N must be a positive integer.')

    if sd1 >= sd2:
        raise ValueError('Sampling limits must satisfy sd1 < sd2.')

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


    validate_search_parameters(
        x_low, x_high, y_low, y_high,
        step_size=step_size,
        N_max=N_max,
        W=W,
        eps=eps
    )
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
    plt.title(f'Trajectories: {name}')
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


    validate_search_parameters(
        x_low, x_high, y_low, y_high,
        step_size=step_size,
        N_max=N_max,
        W=W,
        eps=eps,
        T0=T0,
        alpha=alpha,
        Tmin=Tmin
    )
    
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

#for compareson 

def compare_annealing_schedules(
        f,
        x0, y0,
        x_low, x_high, y_low, y_high,
        step_size,
        schedules,
        seed,
        name
    ):

    all_schedule_runs = {}

    print()
    print(f'Simulated-annealing schedule comparison: {name}')
    print(
        'schedule | run | final value | best value | '
        'attempts | final T | reason'
    )
    print('-' * 95)

    for schedule_number, (schedule_name, T0, alpha) in enumerate(schedules):

        schedule_runs = []

        for run_number in range(6):

            # New generator: independent but reproducible run
            rng_run = np.random.default_rng(
                seed + 1000 * schedule_number + run_number
            )

            result = simulated_annealing(
                x0, y0,
                rng_run,
                f,
                x_low, x_high, y_low, y_high,
                step_size,
                T0,
                alpha,
                Tmin=1e-3,
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

            run = {
                'schedule': schedule_name,
                'T0': T0,
                'alpha': alpha,
                'x0': x0,
                'y0': y0,
                'x_final': x_final,
                'y_final': y_final,
                'z_final': z_final,
                'best_x': best_x,
                'best_y': best_y,
                'best_z': best_z,
                'x_path': x_path,
                'y_path': y_path,
                'z_path': z_path,
                'best_z_path': best_z_path,
                'temperature_values': temperature_values,
                'attempts': attempts,
                'final_temperature': final_temperature,
                'reason': reason
            }

            schedule_runs.append(run)

            print(
                f'{schedule_name:20s} | '
                f'{run_number + 1:3d} | '
                f'{z_final:11.6f} | '
                f'{best_z:10.6f} | '
                f'{attempts:8d} | '
                f'{final_temperature:7.5f} | '
                f'{reason}'
            )

        all_schedule_runs[schedule_name] = schedule_runs

    return all_schedule_runs

#plot
def plot_annealing_schedule_trajectories(
        f,
        x_low, x_high, y_low, y_high,
        representative_runs,
        name
    ):

    x_grid = np.linspace(x_low, x_high, 300)
    y_grid = np.linspace(y_low, y_high, 300)

    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    levels = np.linspace(np.min(Z), np.percentile(Z, 80), 30)

    plt.figure(figsize=(8, 6))

    contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
    plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.4)
    plt.colorbar(contour, label='Function value')

    for i, run in enumerate(representative_runs):

        plt.plot(
            run['x_path'],
            run['y_path'],
            linewidth=1.2,
            label=run['schedule']
        )

        plt.plot(
            run['x0'],
            run['y0'],
            marker='o',
            color='black',
            markersize=6,
            linestyle='none',
            label='Start point' if i == 0 else None
        )

        plt.plot(
            run['x_final'],
            run['y_final'],
            marker='X',
            color='black',
            markersize=7,
            linestyle='none',
            label='Final point' if i == 0 else None
        )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Simulated Annealing Trajectories: {name}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

#plot history
def plot_annealing_schedule_histories(
        representative_runs,
        global_minimum,
        initial_value,
        name,
        shallow_minimum=None
    ):

    plt.figure(figsize=(8, 5))

    for run in representative_runs:

        attempts = np.arange(len(run['z_path']))

        plt.plot(
            attempts,
            run['z_path'],
            label=run['schedule']
        )

    if shallow_minimum is not None:

        plt.axhline(
            shallow_minimum,
            color='tab:red',
            linestyle=':',
            label='Shallow minimum'
        )

    plt.axhline(
        global_minimum,
        color='black',
        linestyle='--',
        label='Global minimum'
    )

    plt.axhline(
        initial_value,
        color='gray',
        linestyle=':',
        label='Initial value'
    )

    plt.xlabel('Attempted step')
    plt.ylabel('Current objective value')
    plt.title(f'Simulated Annealing Objective History: {name}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


# functions for print
def run_annealing_starts(
        f, starts,
        x_low, x_high, y_low, y_high,
        step_size, T0, alpha, seed
    ):

    runs = []

    for i, (x0, y0) in enumerate(starts):

        rng_run = np.random.default_rng(seed + i)

        result = simulated_annealing(
            x0, y0, rng_run, f,
            x_low, x_high, y_low, y_high,
            step_size, T0, alpha, 1e-3,
            N_max=10000, W=100, eps=1e-4
        )

        (
            x_final, y_final, z_final,
            best_x, best_y, best_z,
            x_path, y_path, z_path,
            best_z_path, temperature_values,
            attempts, final_temperature, reason
        ) = result

        runs.append({
            'x0': x0, 'y0': y0,
            'x_final': x_final, 'y_final': y_final,
            'z_final': z_final,
            'best_x': best_x, 'best_y': best_y,
            'best_z': best_z,
            'x_path': x_path, 'y_path': y_path,
            'z_path': z_path,
            'attempts': attempts,
            'final_temperature': final_temperature,
            'reason': reason,
            'step_size': step_size
        })

    return runs


def print_annealing_results(name, runs):

    print()
    print(name)
    print('run | final value | best value | attempts | final T | reason')
    print('-' * 80)

    for i, run in enumerate(runs, start=1):
        print(
            f'{i:3d} | '
            f'{run["z_final"]:11.6f} | '
            f'{run["best_z"]:10.6f} | '
            f'{run["attempts"]:8d} | '
            f'{run["final_temperature"]:7.5f} | '
            f'{run["reason"]}'
        )

#part 5
def hill_climbing_ensemble(
        f, starts,
        x_low, x_high, y_low, y_high,
        step_size, rng
    ):

    best_points = np.empty((len(starts), 2))
    best_values = np.empty(len(starts))

    for i, (x0, y0) in enumerate(starts):

        result = random_move_hill_climbing(
            x0, y0, rng, f,
            x_low, x_high, y_low, y_high,
            step_size,
            N_max=10000,
            W=100,
            eps=1e-4
        )

        x_final, y_final, z_final, _, _, _, _, _ = result

        best_points[i] = [x_final, y_final]
        best_values[i] = z_final

    return best_points, best_values

def annealing_ensemble(
        f, starts,
        x_low, x_high, y_low, y_high,
        step_size, T0, alpha, rng
    ):

    best_points = np.empty((len(starts), 2))
    best_values = np.empty(len(starts))

    for i, (x0, y0) in enumerate(starts):

        result = simulated_annealing(
            x0, y0, rng, f,
            x_low, x_high, y_low, y_high,
            step_size, T0, alpha, 1e-3,
            N_max=10000,
            W=100,
            eps=1e-4
        )

        best_x = result[3]
        best_y = result[4]
        best_z = result[5]

        best_points[i] = [best_x, best_y]
        best_values[i] = best_z

    return best_points, best_values

#heatmaps
def plot_best_point_heatmaps(
        best_point_sets,
        x_low, x_high, y_low, y_high,
        minima,
        name
    ):

    x_edges = np.linspace(x_low, x_high, 31)
    y_edges = np.linspace(y_low, y_high, 31)

    histograms = {}

    for label, points in best_point_sets.items():

        histogram, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=[x_edges, y_edges]
        )

        histograms[label] = histogram

    maximum_count = max(
        np.max(histogram)
        for histogram in histograms.values()
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, (label, histogram) in enumerate(histograms.items()):

        image = axes[i].pcolormesh(
            x_edges,
            y_edges,
            histogram.T,
            cmap='viridis',
            vmin=0,
            vmax=maximum_count,
            shading='auto'
        )

        for x_min, y_min in minima:
            axes[i].plot(
                x_min,
                y_min,
                marker='*',
                color='red',
                markeredgecolor='black',
                markersize=10
            )

        axes[i].set_xlim(x_low, x_high)
        axes[i].set_ylim(y_low, y_high)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_title(label)

        fig.colorbar(image, ax=axes[i], label='Count')

    fig.suptitle(f'Best-Point Locations: {name}')
    plt.tight_layout()
    plt.show()

#best histogramm 
def plot_best_value_histograms(
        best_value_sets,
        global_minimum,
        name
    ):

    all_values = np.concatenate(
        list(best_value_sets.values())
    )

    bins = np.linspace(
        np.min(all_values),
        np.max(all_values),
        50
    )

    plt.figure(figsize=(8, 5))

    for label, values in best_value_sets.items():

        plt.hist(
            values,
            bins=bins,
            density=True,
            histtype='step',
            linewidth=1.8,
            label=label
        )

    plt.axvline(
        global_minimum,
        color='black',
        linestyle='--',
        label='Global minimum'
    )

    plt.xlabel('Best objective value')
    plt.ylabel('Probability density')
    plt.title(f'Distribution of Best Values: {name}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()