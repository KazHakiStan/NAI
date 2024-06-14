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


def generate_neighbor(tour):
    neighbor = tour.copy()
    i, j = random.sample(range(len(tour)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def simulated_annealing(distance_matrix, initial_temperature, cooling_rate, stopping_temperature):
    number_of_cities = len(distance_matrix)

    current_solution = list(range(number_of_cities))
    random.shuffle(current_solution)
    best_solution = current_solution.copy()
    best_distance = calculate_total_distance(best_solution, distance_matrix)

    current_temperature = initial_temperature
    current_distance = best_distance

    solutions = [(current_solution, current_distance)]

    while current_temperature > stopping_temperature:
        neighbor_solution = generate_neighbor(current_solution)
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

        solutions.append((current_solution, current_distance))

    return solutions, best_solution, best_distance


def main():
    file_path = 'tsp_data2.txt'
    distance_matrix = load_tsp_data(file_path)
    initial_temperature = 10000
    cooling_rate = 0.995
    stopping_temperature = 1

    solutions, best_solution, best_distance = simulated_annealing(distance_matrix, initial_temperature, cooling_rate,
                                                                  stopping_temperature)

    for solution, distance in solutions:
        print(f"Solution: {solution}, Distance: {distance}")

    print("\nBest Solution: ", best_solution)
    print("Best Distance: ", best_distance)

    file_path = 'tsp_data.txt'
    distance_matrix = load_tsp_data(file_path)

    solutions, best_solution, best_distance = simulated_annealing(distance_matrix, initial_temperature, cooling_rate,
                                                                  stopping_temperature)

    # for solution, distance in solutions:
        # print(f"Solution: {solution}, Distance: {distance}")

    print("\nBest Solution: ", best_solution)
    print("Best Distance: ", best_distance)


if __name__ == "__main__":
    main()
