import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import random
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD

# Define the square coordinates
min_x, max_x = 5, 50
min_y, max_y = 5, 50

# Sample 3 points uniformly at random within the square
while True:
    points = np.array([
        [random.uniform(min_x, max_x), random.uniform(min_y, max_y)],
        [random.uniform(min_x, max_x), random.uniform(min_y, max_y)],
        [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
    ])

    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    print(eigvals)

    ratio = max(eigvals) / min(eigvals)

    if ratio < 4:
        break

# check if sampled points lie close to the same line
# if yes, resample

# Compute Voronoi diagram
vor = Voronoi(points)

# Plot the square
fig, ax = plt.subplots()
colors = ['blue', 'orange', 'purple']
ls = ['-', '-.', ':']
thickness = [1, 4, 5]

plt.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], lw=2, color='red')

# Plot the sampled points
plt.scatter(points[:, 0], points[:, 1], color=colors)

# Define a predefined length for infinite ridges
predefined_length = 1000

print(vor.vertices[0])


slopes = {
  0: [],
  1: [],
  2: [],
}

# Extract and plot the Voronoi edges
for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    # Infinite ridge
    i = simplex[simplex >= 0][0]  # Finite end of the ridge
    t = points[pointidx[1]] - points[pointidx[0]]  # Direction of the ridge
    t = t / np.linalg.norm(t)
    n = np.array([-t[1], t[0]])  # Normal to the ridge

    far_point = vor.vertices[i] + n * predefined_length
    
    slope = n[1] / n[0]

    pts1_idx = pointidx[0]
    pts2_idx = pointidx[1]

    slopes[pts1_idx].append(slope)
    slopes[pts2_idx].append(slope)

        # plt.plot(
        #   [vor.vertices[i, 0], far_point[0]], 
        #   [vor.vertices[i, 1], far_point[1]], 
        #   'k--')

vor_center = vor.vertices[0]
vor_x = vor_center[0]
vor_y = vor_center[1]
print_slopes = True

margin = 3

for i in slopes.keys():
    slope1, slope2 = slopes[i]
    point = points[i]
    c = colors[i]

    for _ in range(1):
        if print_slopes:
            print(slope1, slope2)
            voronoi_plot_2d(vor, ax=ax)

            pt1_x = 1000
            pt1_y = vor_y - slope1 * (vor_x - pt1_x)
            pt2_x = -1000
            pt2_y = vor_y - slope1 * (vor_x - pt2_x)

            # plt.plot([pt1_x, pt2_x], [pt1_y, pt2_y], color=c, linestyle=ls[i], alpha=0.5, lw=thickness[i])

            pt1_x = 1000
            pt1_y = vor_y - slope2 * (vor_x - pt1_x)
            pt2_x = -1000
            pt2_y = vor_y - slope2 * (vor_x - pt2_x)

            # plt.plot([pt1_x, pt2_x], [pt1_y, pt2_y], color=c, linestyle=ls[i], alpha=0.5, lw=thickness[i])

        prob = LpProblem(f"Optimise Rectangle {i}", LpMaximize)

        x1 = LpVariable("x1", lowBound=min_x + margin, upBound=max_x - margin)
        x2 = LpVariable("x2", lowBound=min_x + margin, upBound=max_x - margin)
        y1 = LpVariable("y1", lowBound=min_y + margin, upBound=max_y - margin)
        y2 = LpVariable("y2", lowBound=min_y + margin, upBound=max_y - margin)

        prob += (x2 - x1), "Objective function"

        r = random.uniform(0.5, 2)
        prob += (x2 - x1) == r * (y2 - y1), "Aspect ratio"

        # constraints on rectangle size
        prob += (x2 - x1) >= 2, "Width constraint x min"
        prob += (y2 - y1) >= 2, "Height constraint y min"
        prob += (x2 - x1) <= 20, "Width constraint x max"
        prob += (y2 - y1) <= 20, "Height constraint y max"

        # slope constraints
        # slope1 
        # check if the current point is above or below the line thorugh vor_x, vor_y with slope1
        slope_vec = np.array([1, slope1])
        slope_vec = slope_vec / np.linalg.norm(slope_vec)

        to_left = True if np.cross(slope_vec, point - vor_center) > 0 else False

        # left, slope > 0 -> minus
        # left, slope < 0 -> plus
        
        # right, slope > 0 -> plus
        # right, slope < 0 -> minus

        if to_left:
            x_margin = margin if slope1 > 0 else -margin
            prob += y1 >= slope1 * (x1 + x_margin - vor_x) + vor_y + margin, "Slope constraint 1"
            prob += y1 >= slope1 * (x2 + x_margin - vor_x) + vor_y + margin, "Slope constraint 2"
            prob += y2 >= slope1 * (x2 + x_margin - vor_x) + vor_y + margin, "Slope constraint 3"
            prob += y2 >= slope1 * (x1 + x_margin - vor_x) + vor_y + margin, "Slope constraint 4"
        else:
            x_margin = -margin if slope1 > 0 else margin
            prob += y1 <= slope1 * (x1 + x_margin - vor_x) + vor_y - margin, "Slope constraint 1"
            prob += y1 <= slope1 * (x2 + x_margin - vor_x) + vor_y - margin, "Slope constraint 2"
            prob += y2 <= slope1 * (x2 + x_margin - vor_x) + vor_y - margin, "Slope constraint 3"
            prob += y2 <= slope1 * (x1 + x_margin - vor_x) + vor_y - margin, "Slope constraint 4"

        # slope2
        # check if the current point is above or below the line thorugh vor_x, vor_y with slope2
        slope_vec = np.array([1, slope2])
        slope_vec = slope_vec / np.linalg.norm(slope_vec)

        to_left = True if np.cross(slope_vec, point - vor_center) > 0 else False

        if to_left:
            x_margin = margin if slope2 > 0 else -margin
            prob += y1 >= slope2 * (x1 + x_margin - vor_x) + vor_y + margin, "Slope constraint 5"
            prob += y1 >= slope2 * (x2 + x_margin - vor_x) + vor_y + margin, "Slope constraint 6"
            prob += y2 >= slope2 * (x2 + x_margin - vor_x) + vor_y + margin, "Slope constraint 7"
            prob += y2 >= slope2 * (x1 + x_margin - vor_x) + vor_y + margin, "Slope constraint 8"
        else:
            x_margin = -margin if slope2 > 0 else margin
            prob += y1 <= slope2 * (x1 + x_margin - vor_x) + vor_y - margin, "Slope constraint 5"
            prob += y1 <= slope2 * (x2 + x_margin - vor_x) + vor_y - margin, "Slope constraint 6"
            prob += y2 <= slope2 * (x2 + x_margin - vor_x) + vor_y - margin, "Slope constraint 7"
            prob += y2 <= slope2 * (x1 + x_margin - vor_x) + vor_y - margin, "Slope constraint 8"

        for j in range(5):
            print("start solving LP")
            prob.solve(PULP_CBC_CMD(msg=False))
            print("Optimal values:")
            for v in prob.variables():
                print(v.name, "=", v.varValue)

            # print rectangle to plt
            sol_x1 = prob.variables()[0].varValue
            sol_x2 = prob.variables()[1].varValue
            sol_y1 = prob.variables()[2].varValue
            sol_y2 = prob.variables()[3].varValue

            plt.plot([sol_x1, sol_x2, sol_x2, sol_x1, sol_x1], [sol_y1, sol_y1, sol_y2, sol_y2, sol_y1], color=c)

        # add constraints to avoid same solution


print(slopes)

# Save the plot
plt.xlim(min_x - 5, max_x + 5)
plt.ylim(min_y - 5, max_y + 5)
plt.savefig("voronoi_polygons.png", dpi=300)
plt.show()