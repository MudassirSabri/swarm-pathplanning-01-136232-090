import random
import math
import numpy as np
import matplotlib.pyplot as plt

SEED = 90
GRID_SIZE = 20
OBSTACLE_COUNT = 70

NUM_PARTICLES = 40
NUM_WAYPOINTS = 12
MAX_ITERATIONS = 200

W_INERTIA = 0.7
C1 = 1.5
C2 = 1.5

random.seed(SEED)
np.random.seed(SEED)


def generate_problem():
    cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]

    start = random.choice(cells)
    goal = random.choice(cells)

    while goal == start:
        goal = random.choice(cells)

    available = [cell for cell in cells if cell != start and cell != goal]
    obstacles = set(random.sample(available, OBSTACLE_COUNT))

    return start, goal, obstacles


def is_free(point, obstacles):
    r, c = point

    if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
        return False

    return point not in obstacles


def line_cells(p1, p2):
    r1, c1 = p1
    r2, c2 = p2

    cells = []

    dr = abs(r2 - r1)
    dc = abs(c2 - c1)

    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1

    err = dr - dc

    r, c = r1, c1

    while True:
        cells.append((r, c))

        if r == r2 and c == c2:
            break

        e2 = 2 * err

        if e2 > -dc:
            err -= dc
            r += sr

        if e2 < dr:
            err += dr
            c += sc

    return cells


def collision_free(path, obstacles):
    for i in range(len(path) - 1):
        segment = line_cells(path[i], path[i + 1])

        for cell in segment:
            if not is_free(cell, obstacles):
                return False

    return True


def build_path(position, start, goal):
    path = [start]

    for i in range(0, len(position), 2):
        row = int(round(position[i]))
        col = int(round(position[i + 1]))

        row = max(0, min(GRID_SIZE - 1, row))
        col = max(0, min(GRID_SIZE - 1, col))

        point = (row, col)

        if point != start and point != goal:
            path.append(point)

    path.append(goal)

    return path


def path_length(path):
    total = 0

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        total += math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)

    return total


def fitness(position, start, goal, obstacles):
    path = build_path(position, start, goal)

    distance = path_length(path)

    if collision_free(path, obstacles):
        return distance

    collision_penalty = 10000

    collision_count = 0

    for i in range(len(path) - 1):
        segment = line_cells(path[i], path[i + 1])

        for cell in segment:
            if cell in obstacles:
                collision_count += 1

    return distance + collision_penalty + (collision_count * 100)


def create_particle():
    position = np.random.uniform(
        0,
        GRID_SIZE - 1,
        NUM_WAYPOINTS * 2
    )

    velocity = np.random.uniform(
        -1,
        1,
        NUM_WAYPOINTS * 2
    )

    return position, velocity


def pso(start, goal, obstacles):
    particles = []

    for _ in range(NUM_PARTICLES):
        position, velocity = create_particle()

        score = fitness(
            position,
            start,
            goal,
            obstacles
        )

        particles.append({
            "position": position,
            "velocity": velocity,
            "score": score,
            "best_position": position.copy(),
            "best_score": score
        })

    global_best = min(
        particles,
        key=lambda particle: particle["score"]
    )

    global_best_position = global_best["best_position"].copy()
    global_best_score = global_best["best_score"]

    history = []

    for iteration in range(MAX_ITERATIONS):

        for particle in particles:

            r1 = np.random.random(len(particle["position"]))
            r2 = np.random.random(len(particle["position"]))

            particle["velocity"] = (
                W_INERTIA * particle["velocity"]
                + C1 * r1 *
                (particle["best_position"] - particle["position"])
                + C2 * r2 *
                (global_best_position - particle["position"])
            )

            particle["position"] += particle["velocity"]

            particle["position"] = np.clip(
                particle["position"],
                0,
                GRID_SIZE - 1
            )

            score = fitness(
                particle["position"],
                start,
                goal,
                obstacles
            )

            particle["score"] = score

            if score < particle["best_score"]:
                particle["best_score"] = score
                particle["best_position"] = particle["position"].copy()

            if score < global_best_score:
                global_best_score = score
                global_best_position = particle["position"].copy()

        history.append(global_best_score)

        if iteration % 20 == 0:
            print(
                f"Iteration {iteration + 1}/{MAX_ITERATIONS} "
                f"- Best Cost: {global_best_score:.2f}"
            )

    return global_best_position, global_best_score, history


def visualize(start, goal, obstacles, path, history):

    plt.figure(figsize=(9, 8))

    obstacle_rows = []
    obstacle_cols = []

    for r, c in obstacles:
        obstacle_rows.append(r)
        obstacle_cols.append(c)

    plt.scatter(
        obstacle_cols,
        obstacle_rows,
        marker="s",
        s=150,
        label="Obstacles"
    )

    path_rows = [point[0] for point in path]
    path_cols = [point[1] for point in path]

    plt.plot(
        path_cols,
        path_rows,
        marker="o",
        linewidth=2,
        label="PSO Path"
    )

    plt.scatter(
        start[1],
        start[0],
        marker="o",
        s=200,
        label="Start"
    )

    plt.scatter(
        goal[1],
        goal[0],
        marker="*",
        s=300,
        label="Goal"
    )

    plt.xlim(-1, GRID_SIZE)
    plt.ylim(-1, GRID_SIZE)

    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))

    plt.grid(True)
    plt.gca().set_aspect("equal")

    plt.title(
        f"PSO Path Planning | Seed = {SEED}"
    )

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.legend()

    plt.savefig(
        "pso_path_result.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(history)

    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")

    plt.title("PSO Convergence")

    plt.grid(True)

    plt.savefig(
        "pso_convergence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def main():

    print("=" * 50)
    print("SWARM INTELLIGENCE - ASSIGNMENT 1")
    print("PSO-Based Path Planning with Obstacles")
    print("=" * 50)

    print(f"Roll Number: 01-136232-090")
    print(f"Random Seed: {SEED}")

    start, goal, obstacles = generate_problem()

    print(f"Grid Size: {GRID_SIZE} x {GRID_SIZE}")
    print(f"Number of Obstacles: {len(obstacles)}")
    print(f"Start Point: {start}")
    print(f"Goal Point: {goal}")

    print("\nRunning PSO...\n")

    best_position, best_score, history = pso(
        start,
        goal,
        obstacles
    )

    best_path = build_path(
        best_position,
        start,
        goal
    )

    valid = collision_free(
        best_path,
        obstacles
    )

    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)

    print(f"Path Found: {valid}")
    print(f"Path Cost: {path_length(best_path):.2f}")
    print(f"Number of Path Points: {len(best_path)}")

    print("\nBest Path:")

    for point in best_path:
        print(point)

    if not valid:
        print("\nPSO did not find a collision-free path.")
        print("Try increasing MAX_ITERATIONS or NUM_PARTICLES.")
    else:
        print("\nCollision-free path successfully found.")

    visualize(
        start,
        goal,
        obstacles,
        best_path,
        history
    )


if __name__ == "__main__":
    main()