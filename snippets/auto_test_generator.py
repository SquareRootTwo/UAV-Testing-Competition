import random
from typing import List
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase
from pulp import *
from scipy.spatial import Voronoi
import numpy as np

DEBUD = True

def score_test(dist: float):
    if dist < 0.25: 
        return 5
    elif dist < 1.0:
        return 2
    elif dist < 1.5:
        return 1
    else:
        return 0
    

class SearchBasedGenerator(object):
    min_size = Obstacle.Size(2, 2, 15)
    max_size = Obstacle.Size(20, 20, 25)
    min_position = Obstacle.Position(-40, 10, 0, 0)
    max_position = Obstacle.Position(30, 40, 0, 90)

    def __init__(self, case_study_file: str) -> None:
        self.case_study = DroneTest.from_yaml(case_study_file)

    def generate(self, budget: int) -> List[TestCase]:
        test_cases = []

        local_budget = 5

        for i in range(budget):
            # parameter type: float
            # parametesrs:          l, w, h, x, y, r
            # parametesrs range:    18,18,10,45,45,90
            # task: search based test cases on these parameters

            # goal: always place 3 obstacles in scene
            #       by first sampling 3 points in the scene
            #       then computing the voronoi diagram of these 3 points 
            #       which divides the scene into 3 parts with 1 vertex and 3 infinite edges.
            #       then sample 3 aspect ratios and optimise for the maximum rectangle in each area 
            #       ->  convex LP problem with size constraints, aspect ratio constraints, 
            #           slope constraints and region boundary constraints

            search_space = []

            li = 0
            while li < local_budget:
                # TODO: avoid case where all points lie on a line -> voronoi vertex is at infinity 
                points = np.array([
                    [random.uniform(self.min_position.x, self.max_position.x), random.uniform(self.min_position.y, self.max_position.y)],
                    [random.uniform(self.min_position.x, self.max_position.x), random.uniform(self.min_position.y, self.max_position.y)],
                    [random.uniform(self.min_position.x, self.max_position.x), random.uniform(self.min_position.y, self.max_position.y)]
                ])

                vor = Voronoi(points)
                # since we only sample 3 points, we have always 1 vertex
                vor_center = vor.vertices[0]
                vor_x, vor_y = vor_center

                slopes = {0: [], 1: [], 2: []}

                if DEBUD:
                    import matplotlib.pyplot as plt
                    plt.clf()
                    fig, ax = plt.subplots()
                    
                    # line styles
                    colors = ['blue', 'orange', 'purple']
                    ls = ['-', '-.', ':']
                    thickness = [1, 4, 5]

                    plt.plot(vor_x, vor_y, 'ro', color='red')
                    
                    # area plot
                    plt.plot(
                        [self.min_position.x, self.max_position.x, self.max_position.x, self.min_position.x, self.min_position.x], 
                        [self.min_position.y, self.min_position.y, self.max_position.y, self.max_position.y, self.min_position.y], 
                        lw=2, 
                        color='red'
                    )
                    plt.scatter(points[:, 0], points[:, 1], color=colors)


                for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
                    simplex = np.asarray(simplex)
                    # infinite ridge -> compute finite end of the ridge
                    i = simplex[simplex >= 0][0]

                    # direction of the ridge
                    t = points[pointidx[1]] - points[pointidx[0]]  
                    t = t / np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])

                    # far_point = vor.vertices[i] + n * predefined_length
                    
                    slope = n[1] / n[0]

                    pts1_idx = pointidx[0]
                    pts2_idx = pointidx[1]

                    slopes[pts1_idx].append(slope)
                    slopes[pts2_idx].append(slope)
                
                obstacles = []

                for s_i in slopes.keys():
                    point = points[s_i]
                    slope1, slope2 = slopes[s_i]

                    # LP problem
                    prob = LpProblem(f"optimise_rectangle_{s_i}", LpMaximize)

                    # sample aspect ratio
                    r = random.uniform(0.5, 2)

                    x1 = LpVariable("x1", lowBound=self.min_position.x, upBound=self.max_position.x)
                    x2 = LpVariable("x2", lowBound=self.min_position.x, upBound=self.max_position.x)
                    y1 = LpVariable("y1", lowBound=self.min_position.y, upBound=self.max_position.y)
                    y2 = LpVariable("y2", lowBound=self.min_position.y, upBound=self.max_position.y)

                    prob += (x2 - x1), "objective_function"

                    # aspect ratio constraint
                    prob += (x2 - x1) == r * (y2 - y1), "aspect_ratio_constraint"

                    # constraints on rectangle size
                    prob += (x2 - x1) >= self.min_size.w, "width_x_min_constraint"
                    prob += (y2 - y1) >= self.min_size.l, "length_y_min_constraint"
                    prob += (x2 - x1) <= self.max_size.w, "width_x_max_constraint"
                    prob += (y2 - y1) <= self.max_size.l, "length_y_max_constraint"

                    # slope constraints
                    # slope1 
                    # check if the current point is above or below the line thorugh vor_x, vor_y with slope1
                    slope_vec = np.array([1, slope1])
                    slope_vec = slope_vec / np.linalg.norm(slope_vec)

                    # this is only to get the inequality sign for the constraint (either <= or >=)
                    to_left = True if np.cross(slope_vec, point - vor_center) > 0 else False

                    if to_left:
                        prob += y1 >= slope1 * (x1 - vor_x) + vor_y, "slope_constraint_1"
                        prob += y1 >= slope1 * (x2 - vor_x) + vor_y, "slope_constraint_2"
                        prob += y2 >= slope1 * (x2 - vor_x) + vor_y, "slope_constraint_3"
                        prob += y2 >= slope1 * (x1 - vor_x) + vor_y, "slope_constraint_4"
                    else:
                        prob += y1 <= slope1 * (x1 - vor_x) + vor_y, "slope_constraint_1"
                        prob += y1 <= slope1 * (x2 - vor_x) + vor_y, "slope_constraint_2"
                        prob += y2 <= slope1 * (x2 - vor_x) + vor_y, "slope_constraint_3"
                        prob += y2 <= slope1 * (x1 - vor_x) + vor_y, "slope_constraint_4"

                    # slope2
                    # check if the current point is above or below the line thorugh vor_x, vor_y with slope2
                    slope_vec = np.array([1, slope2])
                    slope_vec = slope_vec / np.linalg.norm(slope_vec)

                    # this is only to get the inequality sign for the constraint (either <= or >=)
                    to_left = True if np.cross(slope_vec, point - vor_center) > 0 else False

                    if to_left:
                        prob += y1 >= slope2 * (x1 - vor_x) + vor_y, "slope_constraint_5"
                        prob += y1 >= slope2 * (x2 - vor_x) + vor_y, "slope_constraint_6"
                        prob += y2 >= slope2 * (x2 - vor_x) + vor_y, "slope_constraint_7"
                        prob += y2 >= slope2 * (x1 - vor_x) + vor_y, "slope_constraint_8"
                    else:
                        prob += y1 <= slope2 * (x1 - vor_x) + vor_y, "slope_constraint_5"
                        prob += y1 <= slope2 * (x2 - vor_x) + vor_y, "slope_constraint_6"
                        prob += y2 <= slope2 * (x2 - vor_x) + vor_y, "slope_constraint_7"
                        prob += y2 <= slope2 * (x1 - vor_x) + vor_y, "slope_constraint_8"

                    try:
                        prob.solve(PULP_CBC_CMD(msg=False))
                        sol_x1 = prob.variables()[0].varValue
                        sol_x2 = prob.variables()[1].varValue
                        sol_y1 = prob.variables()[2].varValue
                        sol_y2 = prob.variables()[3].varValue

                        if DEBUD:
                            # plot voronoi center
                            plt.plot([sol_x1, sol_x2, sol_x2, sol_x1, sol_x1], [sol_y1, sol_y1, sol_y2, sol_y2, sol_y1], color="red")

                            # plot voronoi edges
                            pt1_x = 1000
                            pt1_y = vor_y - slope1 * (vor_x - pt1_x)
                            pt2_x = -1000
                            pt2_y = vor_y - slope1 * (vor_x - pt2_x)

                            plt.plot([pt1_x, pt2_x], [pt1_y, pt2_y], color=c, linestyle=ls[i], alpha=0.5, lw=thickness[i])

                            pt1_x = 1000
                            pt1_y = vor_y - slope2 * (vor_x - pt1_x)
                            pt2_x = -1000
                            pt2_y = vor_y - slope2 * (vor_x - pt2_x)

                            plt.plot([pt1_x, pt2_x], [pt1_y, pt2_y], color=c, linestyle=ls[i], alpha=0.5, lw=thickness[i])
                        
                        center = Obstacle.Position(
                            x=(sol_x1 + sol_x2) / 2, 
                            y=(sol_y1 + sol_y2) / 2, 
                            z=0, 
                            r=0, # this is the downside of the LP approach, we do not have a rotation
                        )
                        size = Obstacle.Size(
                            l=sol_x2 - sol_x1, 
                            w=sol_y2 - sol_y1,
                            h=random.uniform(self.min_size.h, self.max_size.h),
                        )

                        obstacle = Obstacle(size, center)
                        obstacles.append(obstacle)
                    except:
                        print("Exception during LP solving, skipping the test")
                        continue

                if DEBUD:
                    plt.savefig(f"/src/generator/results/{i}_{li}_obstacle_placement.png")

                try:
                    test = TestCase(self.case_study, obstacles)
                    test.execute()
                    distances = test.get_distances()
                    min_dist = min(distances)
                    score = score_test(min_dist)

                    print(f"minimum_distance:{min_dist}, score: {score}")

                    # TODO: only return soft and hard fail test cases
                    # hard fail: min_dist == 0
                    # soft fail: min_dist < 1.5

                    test.plot()
                    search_space.append((test, score))

                    li += 1

                except Exception as e:
                    print("Exception during test execution, skipping the test")
                    print(e)

            # rank the test cases based on the score and return the best one
            search_space.sort(key=lambda x: x[1], reverse=True)
            test_cases.append(search_space[0][0])

        ### You should only return the test cases
        ### that are needed for evaluation (failing or challenging ones)

        return test_cases


if __name__ == "__main__":
    generator = RandomGenerator("case_studies/mission1.yaml")
    generator.generate(3)
