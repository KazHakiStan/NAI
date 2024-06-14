import numpy as np
import random


def load_tsp_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(int, line.split())) for line in lines]
    return matrix


def calculate_total_distance(tour, distance_matrix):
    total_distance = 0
    number_of_cities = len(tour)
    for i in range(number_of_cities):
        total_distance += distance_matrix[tour[i]][tour[(i + 1) % number_of_cities]]
    return total_distance


def generate_neighbor(tour, counter):
    neighbor = tour.copy()
    if counter == len(tour) - 1:
        i = counter
        j = counter - 1
    else:
        i = counter
        j = counter + 1
    # print("i " + str(i) + " j " + str(j))  
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def simulated_annealing(distance_matrix, initial_temperature, cooling_rate, stopping_temperature):
    number_of_cities = len(distance_matrix)

    current_solution = list(range(number_of_cities))
    print("here", current_solution)
    random.shuffle(current_solution)
    best_solution = current_solution.copy()
    best_distance = calculate_total_distance(best_solution, distance_matrix)

    current_temperature = initial_temperature
    current_distance = best_distance

    solutions = [(current_solution, current_distance)]

    counter = 0

    while current_temperature > stopping_temperature:
        if counter == len(current_solution): 
            counter = 0
        neighbor_solution = generate_neighbor(current_solution, counter)
        neighbor_distance = calculate_total_distance(neighbor_solution, distance_matrix)

        if neighbor_distance < current_distance:
            accept_probability = 1
        else:
            accept_probability = np.exp((current_distance - neighbor_distance) / current_temperature)

        if accept_probability > random.random():
            current_solution = neighbor_solution
            current_distance = neighbor_distance

            if current_distance < best_distance:
                best_solution = current_solution
                best_distance = current_distance

        current_temperature *= cooling_rate
        counter += 1
        solutions.append((current_solution, current_distance))

    return solutions, best_solution, best_distance


def main():
    file_path = 'tsp_data2.txt'
    distance_matrix = load_tsp_data(file_path)
    initial_temperatures = [5000, 7000, 10000]
    cooling_rates = [0.8, 0.995, 0.999]
    stopping_temperatures = [1, 10, 50]

    best_hyperparameters = None
    best_training_distance = float('inf')

    for initial_temperature in initial_temperatures:
        for cooling_rate in cooling_rates:
            for stopping_temperature in stopping_temperatures:
                solutions, best_solution, best_distance = simulated_annealing(
                    distance_matrix,
                    initial_temperature,
                    cooling_rate,
                    stopping_temperature
                )
                if best_distance < best_training_distance:
                    best_training_distance = best_distance
                    best_hyperparameters = (initial_temperature, cooling_rate, stopping_temperature)

    print("Best Hyperparameters: ", best_hyperparameters)
    print("Best Training Distance: ", best_training_distance)

    # for solution, distance in solutions:
    #     print(f"Solution: {solution}, Distance: {distance}")

    print("\nBest Solution: ", best_solution)
    print("Best Distance: ", best_distance)

    file_path = 'tsp_data.txt'
    distance_matrix = load_tsp_data(file_path)

    best_initial_temperature, best_cooling_rate, best_stopping_temperature = best_hyperparameters
    solutions, best_solution, best_distance = simulated_annealing(
        distance_matrix,
        best_initial_temperature,
        best_cooling_rate,
        best_stopping_temperature
    )

    for solution, distance in solutions:
        print(f"Solution: {solution}, Distance: {distance}")

    print("\nBest Solution: ", best_solution)
    print("Best Distance: ", best_distance)


if __name__ == "__main__":
    main()
