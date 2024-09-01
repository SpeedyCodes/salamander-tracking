"""
Rune De Coninck

This document will use an algorithm [Edward J. Groth 1986: A pattern-matching algorithm for two-dimensional coordinate
lists], [Journal of Applied Ecology 2005: An astronomical pattern-matching algorithm for computer-aided identification
of whale sharks Rhincodon typus] to compare a list of coordinates (which represents dots of the salamanders).
Thus, the goal is to match or not match certain patterns of two salamanders and thus, tell if we are working with the
same salamander or not.

This is done in several steps:
0. Loading in the list of coordinates/points.
1. Check if the number of coordinates/points is approximately equal.
2. Selecting the points to be matched.
3. Generating lists of triangles.
4. Matching the triangles.
5. Reducing the number of false matches.
6. Assigning matched points through voting.
7. Computing a score for each matching point patterns.
8. Displaying the best matches to the user.
"""

from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import copy
from time import time
import threading
from typing import List, Tuple, Set


def compare_dot_patterns(unknown_image_coordinates: Set[Tuple[float, float]],
                         database_of_coordinates: List[Tuple[str, Set[Tuple[float, float]]]],
                         tol: float = 0.01, is_second_run: bool = True, plot_r_values: bool = False,
                         thread_count: int = 1):
    """ This function will include a bunch of other functions ...

    INPUT: unknown_image_coordinates contains the coordinates of the image we want to compare with our database.
    INPUT: database_of_coordinates is a list of tuples(name, coords) that represents all the items of our database.
    INPUT: the tolerance tol is a parameter that the user is free to set. If detected dots have distance less than
    a scaler multiplied by tol, then we consider these dots as one. (We remove one of the two dots, this ensures that
    the numerical computations are stable).
    INPUT: is_second_run is True when we go twice through the algorithm, if it is False we go one time through the
    algorithm. A second run ensures that the matches are of higher quality, but at the cost of a higher chance of
    missing True matches.
    INPUT: plot_r_values is a boolean that determines whether to plot the r_values in a histogram.
    OUTPUT: a sorted list of scores. """

    loading_bar = tqdm(total=len(database_of_coordinates) + 1)  # Keeps track on how far the algorithm is.

    list_of_scores = []  # Will include all the images in the database and their matching score with the unknown image.

    # Select the points to be used from the coordinates of the unknown image
    list_coordinates = select_points_to_be_matched(unknown_image_coordinates, tol=3 * tol)

    start_time = time()

    def matching_procedure(name_image_from_database, list_coordinates_image_from_database):
        list_coordinates_image_from_database = select_points_to_be_matched(list_coordinates_image_from_database, 3 * tol)
        matched_points, V, f_t, V_max = run_through_algorithm(list_coordinates, list_coordinates_image_from_database,
                                                              tol=tol,
                                                              plot_r_values=plot_r_values)

        if is_second_run:
            # Go a second time through the algorithm, using the output of the first run.
            list_coordinates_new = [points[0] for points in matched_points]
            list_coordinates_image_database_new = [points[1] for points in matched_points]

            matched_points, V, f_t, V_max = run_through_algorithm(list_coordinates_new,
                                                                  list_coordinates_image_database_new,
                                                                  tol=tol, plot_r_values=plot_r_values)

        # Compute a score, based on V and f_t and a second score based on the number of successful matched points.
        if len(list_coordinates) != 0:
            S2 = len(matched_points) / len(list_coordinates)
        else:
            S2 = 0
        S1, S1_rel, keyword = compute_score(V, f_t, V_max, S2)

        # Add the result to the list of scores.
        list_of_scores.append((S1, S1_rel, S2, keyword, name_image_from_database))

    threads = []

    # Start matching procedure for every image in the database.
    if thread_count == 1:  # no threading
        for name_image_database, list_coordinates_image_database in database_of_coordinates:
            matching_procedure(name_image_database, list_coordinates_image_database)
            loading_bar.update(1)
    else:  # threaded execution
        def thread_function(start, end):
            for j in range(start, end):
                matching_procedure(database_of_coordinates[j][0], database_of_coordinates[j][1])
                loading_bar.update(1)

        for i in range(thread_count):
            threads.append(threading.Thread(target=thread_function, args=(
                i * len(database_of_coordinates) // thread_count,
                (i + 1) * len(database_of_coordinates) // thread_count)))
            threads[-1].start()

        for thread in threads:
            thread.join()

    print(f"Time for matching procedure: {time() - start_time} seconds ---")

    # Sort the list_of_scores based on highest combined score.
    weight_S1 = 0.3
    weight_S2 = 0.7
    list_of_scores = sort_list_of_scores(list_of_scores, weight_S1, weight_S2)

    return list_of_scores


def run_through_algorithm(list_coordinates, list_coordinates_image_database, tol: float,
                          plot_r_values: bool):
    """ This method runs through all the steps of the algorithm, from creating triangles to assigning matched points."""

    # Check the number of detected points.
    if not check_number_of_points(list_coordinates, list_coordinates_image_database):
        return [], 0, 0.0, 0  # This is not a match, go to the next pattern in the database.

    if len(list_coordinates) == 0 or len(list_coordinates_image_database) == 0:
        return [], 0, 0.0, 0

    # Generate a list of triangles and some extra information.
    list_triangles, r_values = generate_list_triangles(list_coordinates, tol, R_max=8, C_max=0.99, s_max=0.85)
    # We have found R_max by experimentally looking at the histogram with R-values.

    if plot_r_values:
        # Plotting the histogram of r_values.
        histogram_r_values(r_values)

    # Make a new list with a lot of triangles and some extra information.
    list_triangles_database, _ = generate_list_triangles(list_coordinates_image_database, tol, R_max=8,
                                                         C_max=0.99,
                                                         s_max=0.85)

    # Calculating matches.
    matches_triangles = matching_triangles(list_triangles, list_triangles_database)

    # Reducing false matches.
    matches_triangles = reduce_false_matches(matches_triangles)

    V_max = 3 * len(matches_triangles)

    # Matching the points.
    matched_points, V, f_t = assign_matched_points(matches=matches_triangles,
                                                   total_nr_triangles_before_matching=len(list_triangles))

    return matched_points, V, f_t, V_max


""" STEP 1: Selecting the points to be matched."""


def select_points_to_be_matched(list_coordinates, tol: float):
    """ This method will select the points that we can use for the remainder of the algorithm.
    If two points are closer than a distance tol of each other, we will remove one of the two points and consider them
    to be the same point. This ensures good numerical calculations."""

    selected_points = set()

    for point in list_coordinates:
        # Check if the point is too close to any of the already selected points.
        is_close = False
        for selected in selected_points:
            distance = math.dist(point, selected)
            if distance < tol:
                is_close = True
                break
        # If not close, add the point to the selected points list.
        if not is_close:
            selected_points.add(point)

    return selected_points


""" STEP 2: Check if the number of coordinates/points is approximately equal. """


def check_number_of_points(list_coordinates1, list_coordinates2, tol: int = 10):
    """ This method returns True if the difference of number of points is less than the tolerance and False otherwise.
    In this way, a salamander with a lot of points won't get matched with a salamander with few points.
    Furthermore, this ensures that the lists of coordinates are approximately of equal size. From the literature,
    we know that this ensures good results."""

    return abs(len(list_coordinates1) - len(list_coordinates2)) <= tol


""" STEP 3: Generating lists of triangles. """


def orientation_triangle(vertex1, vertex2, vertex3) -> str:
    """ Determines the orientation of a triangle, it determines in which way we can traverse through vertex1,
    vertex2, and vertex3. This is based on the Shoelace formula."""
    (x1, y1) = vertex1
    (x2, y2) = vertex2
    (x3, y3) = vertex3

    A = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

    if A > 0.001:
        return "Counter-clockwise"
    elif A < - 0.001:
        return "Clockwise"
    else:
        return "Collinear"


def generate_list_triangles(list_coordinates, tol, R_max, C_max, s_max):
    """ This method will generate all possible triangles between every three distinct points in the list of coordinates.
    We return a big list of all the triangles and extra info such as ratio of sides, perimeter, orientation,
    different tolerances, cosine of a certain angle. Detailed info can be found in this code as comments.

    OUTPUT list_triangles: [ ( (vertex1, vertex2, vertex3), R, C, tol_r, tol_c, log_p, orientation ) ]. """

    """ First try to compute r3_max, this is the biggest side of all possible triangles, we do this by creating all 
    possible triangles between every three points in the list of coordinates. """
    temp_data = []
    r3_max = 0

    for temp1, temp2, temp3 in itertools.combinations(list_coordinates, 3):
        # Calculate the lengths of the sides of the triangle
        temp12 = math.dist(temp1, temp2)
        temp23 = math.dist(temp2, temp3)
        temp31 = math.dist(temp3, temp1)

        longest_side_length = max(temp12, temp23, temp31)

        if longest_side_length > r3_max:
            r3_max = longest_side_length

        temp_data.append((temp1, temp2, temp3, temp12, temp23, temp31))

    list_triangles = []
    r_values = []

    """ Second, collect the data from the first step."""
    for temp1, temp2, temp3, temp12, temp23, temp31 in temp_data:

        """ Third, we want to organize the vertices ( = points) and the lengths of the sides in the following way.
        We name the vertices and sides such that the following holds:
        The shortest side is defined to lie between vertices 1 and 2, the intermediate side between vertices 2 and 3,
        and the longest side between vertices 1 and 3. """

        sides = [(temp12, (temp1, temp2)), (temp23, (temp2, temp3)), (temp31, (temp3, temp1))]
        sides.sort()  # Sort the sides such that the smallest side is the first element.
        shortest_side, intermediate_side, longest_side = sides

        # Update r3_max to the current longest side if it's larger.
        r3 = longest_side[0]
        if r3 > r3_max:
            r3_max = r3

        # Shortest side must be between vertex 1 and 2.
        vertex1, vertex2 = shortest_side[1]

        # Longest side must be between vertex 1 and 3, so we need to swap if needed.
        if vertex1 not in longest_side[1]:
            vertex1, vertex2 = vertex2, vertex1

        # Assign vertex 3 to the remaining point.
        vertex3 = next(vertex for vertex in [temp1, temp2, temp3] if vertex not in (vertex1, vertex2))

        # Ensure the longest side is between vertex1 and vertex3, otherwise swap the vertices.
        if (vertex1, vertex3) != longest_side[1] and (vertex3, vertex1) != longest_side[1]:
            vertex2, vertex3 = vertex3, vertex2

        """ Fourth, we compute a lot of information about the vertices and sides. """
        # Coordinates of the vertices.
        (x1, y1) = vertex1
        (x2, y2) = vertex2
        (x3, y3) = vertex3

        # Orientation of the triangle, how can we traverse through vertex1, vertex2 and vertex3.
        orientation = orientation_triangle(vertex1, vertex2, vertex3)
        if orientation == 'Collinear':
            continue  # Since this will not be a triangle.

        # Ratio of the longest to the shortest side.
        r2 = shortest_side[0]
        r3 = longest_side[0]
        R = r3 / r2
        if R > R_max:
            continue  # Since otherwise, this will result in triangles with to big tol_r and hence they will match
            # with a lot of triangles that are not supposed to be matches.

        # With r_values, we can see how the R values are distributed.
        r_values.append(R)

        # Ratio of the longest side to the maximum of the longest sides.
        s = r3 / r3_max
        if s > s_max:
            continue  # Image quality and projection effect distortion are largest for the big triangles.
            # This gives more problems when trying to match the triangles.

        # Sine and Cosine of the angle at vertex 1.
        C = (1 / (r3 * r2)) * ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1))
        if C > C_max:
            continue  # In this case, the triangle will almost be a line and thus, our algorithm will be more
            # susceptible to generate false matches between all of these kind of "fake triangles".
        S = (1 / (r3 * r2)) * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))

        # Tolerances in R and C.
        F = (tol ** 2) * ((1 / (r3 ** 2)) - (C / (r3 * r2)) + (1 / (r2 ** 2)))
        tol_r = math.sqrt(2 * R * R * F)
        tol_c = math.sqrt(2 * S * S * F + 3 * C * C * F * F)

        # Logarithm of the perimeter.
        log_p = math.log(shortest_side[0] + intermediate_side[0] + longest_side[0])

        """ Fifth, add the information to list_triangles. """
        list_triangles.append(((vertex1, vertex2, vertex3), R, C, tol_r, tol_c, log_p, orientation))

    return list_triangles, r_values


def histogram_r_values(r_values):
    """ This method generates a histogram of the r_values. """

    plt.figure(figsize=(8, 6))
    plt.hist(r_values, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of R Values')
    plt.xlabel('R Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return None


""" STEP 4: Matching the triangles. """


def is_possible_match(triangle1, triangle2) -> bool:
    """ This method returns True if there could be a potential match between the two triangles. Otherwise, it will
    return False. We use the fact that similar triangles (up to orientation) are uniquely determined by a length ratio R
    and the cosine of an angle C. We use the tolerances computed in step 3."""

    (_, R_a, C_a, tol_r_a, tol_c_a, _, _) = triangle1
    (_, R_b, C_b, tol_r_b, tol_c_b, _, _) = triangle2

    check1 = (R_a - R_b) ** 2 < (tol_r_a ** 2) + (tol_r_b ** 2)
    check2 = (C_a - C_b) ** 2 < (tol_c_a ** 2) + (tol_c_b ** 2)

    return check1 and check2


def matching_triangles_old(list_triangles1, list_triangles2):
    """ This method tries to find a match between two lists of triangles. Essentially we can just compare all the
    triangles of both lists and use is_possible_match on them, but for optimal computational time, we will do something
    like a merge sort. By sorting the lists and using tolerance ranges, we significantly reduce the number of
    comparisons compared to a brute force approach. Instead of checking all pairs, we only check triangles within a
    specific range, guided by the maximum tolerances.

    INPUT: list_triangles1: this is the list of triangles, determined by the unknown image.
    INPUT: list_triangles2: this is the list of triangles, determined by some image from our database.

    CONTENT: list_triangles: [ ( (vertex1, vertex2, vertex3), R, C, tol_r, tol_c, log_p, orientation ) ] """

    if len(list_triangles1) == 0 or len(list_triangles2) == 0:
        return []

    # First sort the two lists, based on increasing R values.
    list_triangles1.sort(key=lambda x: x[1])
    list_triangles2.sort(key=lambda x: x[1])

    # Then, find the maximal tol_r for both lists.
    max_tol_r_a = max([tol_r_a for _, _, _, tol_r_a, _, _, _ in list_triangles1])
    max_tol_r_b = max([tol_r_b for _, _, _, tol_r_b, _, _, _ in list_triangles2])

    matches = []
    # We iterate over list 2 with variable j, variable "i" also iterates over list 2, but it finds the lower bound for
    # j. Then the parameter j will start at this lower bound and end at an upper bound. This prevents brute force
    # calculation and allows us to only compare some of the triangles.
    i = 0

    for triangle1 in list_triangles1:
        R_a = triangle1[1]

        # Determine the range of indices in list 2 to consider, the bounds are found by using the triangle inequality at
        # the conditions of the function is_possible_match.
        while i < len(list_triangles2) and list_triangles2[i][1] < R_a - (max_tol_r_a + max_tol_r_b):
            i += 1

        j = i  # Start at the lower bound.

        # Temporary list to store possible matches for the current triangle from list 1
        potential_matches = []

        while j < len(list_triangles2) and list_triangles2[j][1] <= R_a + (max_tol_r_a + max_tol_r_b):
            triangle2 = list_triangles2[j]

            # Check if this pair of triangles match.
            if is_possible_match(triangle1, triangle2):
                potential_matches.append((triangle1, triangle2))

            j += 1

        # If there are any potential matches, find the closest one based on the check1 criteria from the previous
        # method. This ensures that we have at most one match for every triangle of list 1.
        if len(potential_matches) > 0:
            best_match = min(potential_matches, key=lambda x: (R_a - x[1][1]) ** 2)
            matches.append(best_match)

    return matches


def matching_triangles(list_triangles1, list_triangles2) -> set:
    """ This method tries to find a match between two lists of triangles. Essentially we can just compare all the
    triangles of both lists and use is_possible_match on them, but for optimal computational time, we will first sort
    all the triangles of the second list. We sort them in such a way that we first try the most likely matches
    [eq 17 Journal Applied Ecology]. When a match is found, we move on to the next triangle from list 1 since we only
    want 1 match (the best match) for every triangle from list 1.

    INPUT: list_triangles1: this is the list of triangles, determined by the unknown image.
    INPUT: list_triangles2: this is the list of triangles, determined by some image from our database.

    CONTENT: list_triangles: [ ( (vertex1, vertex2, vertex3), R, C, tol_r, tol_c, log_p, orientation ) ] """

    if len(list_triangles1) == 0 or len(list_triangles2) == 0:
        return set()

    matches = set()

    for triangle1 in list_triangles1:
        R_a = triangle1[1]
        """
        [eq 17 Journal Applied Ecology], not chosen to work with since it dramatically increases computation time.
        But code is the following:
        C_a = triangle1[2]
        tol_r_a = triangle1[3]
        tol_c_a = triangle1[4]

        # Sort the triangles from list 2, [eq 17 Journal Applied Ecology].
        list_triangles2.sort(key=lambda triangle: (((R_a - triangle[1]) ** 2) / (tol_r_a ** 2 + triangle[3] ** 2)) +
                                                  (((C_a - triangle[2]) ** 2) / (tol_c_a ** 2 + triangle[4] ** 2)))
        """
        list_triangles2.sort(key=lambda x: (R_a - x[1]) ** 2)  # Quicker and still reliable way to sort list 2.

        for triangle2 in list_triangles2:
            # Check if this pair of triangles match.
            if is_possible_match(triangle1, triangle2):
                matches.add((triangle1, triangle2))
                break  # Go to the next triangle, since we have a match for this triangle.

    return matches


""" STEP 5: Reducing the number of false matches. """


def compute_extra_info_for_matches(matches):
    """ This method calculates the logarithm of the magnification factor M and the same- or opposite-sense variable
    (triangles or same sense if they are both the same oriented, otherwise they are opposite-sense) for each match.
    Then it returns the matches, including the extra info for every match.

    INPUT: a match is a tuple of two triangles, such a triangle is the following:
    ( (vertex1, vertex2, vertex3), R, C, tol_r, tol_c, log_p, orientation )."""

    matches_extra_info = set()

    for (triangle1, triangle2) in matches:
        log_p1 = triangle1[5]
        log_p2 = triangle2[5]

        log_M = log_p1 - log_p2  # M is the magnification factor between the triangles (the scaling for in- and
        # out zooming).

        if triangle1[6] == triangle2[6]:  # Compare the orientation of the triangles.
            sense = 'same'
        else:
            sense = 'opposite'

        matches_extra_info.add((triangle1, triangle2, log_M, sense))

    return matches_extra_info


def reduce_false_matches(matches: set[tuple], iteration_limit: bool = 20, factor: float = 0.1) -> set:
    """ This method reduces the number of false matches based on the logarithm of the magnification factor M,
    its distribution and the orientation of the triangles."""

    """ First, calculate the logarithm of the magnification factor M and the same- or opposite-sense variable 
    (triangles or same sense if they are both the same oriented, otherwise they are opposite-sense) for each match."""

    matches: set = compute_extra_info_for_matches(matches)

    """ matches is of the following type:
    (triangle1, triangle2, log_M, sense), where a triangle is of the following type:
    ( (vertex1, vertex2, vertex3), R, C, tol_r, tol_c, log_p, orientation ). """

    """ Second, we will do some iterations to filter the matches, based on the log_M values.
    We can do this since all the true matches will have approximately the same log_M, while the false
    matches will have distinct log_M values."""

    is_match_discarded = True
    iteration = 0

    if len(matches) == 0:
        return set()

    while is_match_discarded and iteration < iteration_limit and len(matches) > 0:

        # Do some initialising calculations:
        log_M_list = [x[2] for x in matches]

        n_plus = sum(1 for _, _, _, sense in matches if sense == 'same')  # Number of same senses.
        n_minus = sum(1 for _, _, _, sense in matches if sense == 'opposite')  # Number of opposite senses.

        m_true = abs(n_plus - n_minus)  # Estimate number of true matches.
        m_false = n_plus + n_minus - m_true  # Estimate number of false matches.

        mean_log_M = np.mean(log_M_list)
        std_dev_log_M = np.std(log_M_list)

        if m_false > m_true:
            scaler = 1
        elif factor * m_true > m_false:
            scaler = 3
        else:
            scaler = 2

        is_match_discarded = False
        matches_copy = copy.copy(matches)

        # Only keep the matches that are close to the mean of log_M, this ensures we keep the true matches.
        for (triangle1, triangle2, log_M, sense) in matches_copy:
            if abs(log_M - mean_log_M) <= scaler * std_dev_log_M:
                continue
            else:
                is_match_discarded = True
                matches.remove((triangle1, triangle2, log_M, sense))

        iteration += 1

    """ Third, we filter on the sense values. So after the iterative proces from the previous step, 
    we know only keep the matches that have same or opposite sense, based on which version is the most common."""

    n_plus = sum(1 for _, _, _, sense in matches if sense == 'same')  # Number of same senses.
    n_minus = sum(1 for _, _, _, sense in matches if sense == 'opposite')  # Number of opposite senses.

    matches_copy = copy.copy(matches)

    if n_plus >= n_minus:
        remove_matches = 'opposite'
    else:
        remove_matches = 'same'

    for (triangle1, triangle2, log_M, sense) in matches_copy:
        if sense == remove_matches:
            matches.remove((triangle1, triangle2, log_M, sense))

    return matches


""" STEP 6: Assigning matched points through voting. """


def assign_matched_points(matches: set[tuple], total_nr_triangles_before_matching: int):
    """ This method assigns points to each other using a voting system. For every matching two triangles, we cast
    three votes for the corresponding vertices. Then we only keep the most frequent votes and these are most likely
    true matching points."""

    if len(matches) == 0 or total_nr_triangles_before_matching == 0:
        return matches, 0, 0.0

    contributing_triangles = set()  # Tracks the unique triangles contributing to the voting proces.

    # First cast all the votes for every matching triangle.
    voting_dict = dict()
    for triangle1, triangle2, _, _ in matches:
        (vertex11, vertex12, vertex13), _, _, _, _, _, _ = triangle1
        (vertex21, vertex22, vertex23), _, _, _, _, _, _ = triangle2

        contributing_triangles.add(triangle1)

        if (vertex11, vertex21) not in voting_dict.keys():
            voting_dict[(vertex11, vertex21)] = 1
        else:
            voting_dict[(vertex11, vertex21)] += 1

        if (vertex12, vertex22) not in voting_dict.keys():
            voting_dict[(vertex12, vertex22)] = 1
        else:
            voting_dict[(vertex12, vertex22)] += 1

        if (vertex13, vertex23) not in voting_dict.keys():
            voting_dict[(vertex13, vertex23)] = 1
        else:
            voting_dict[(vertex13, vertex23)] += 1

    # Convert the dictionary to a list and sort the votes in decreasing order.
    voting_list = list(voting_dict.items())
    """ Voting list is of the form [ ( (vertex, vertex), vote ), ... ]"""
    voting_list.sort(key=lambda item: item[1], reverse=True)

    nr_of_most_votes = voting_list[0][1]
    threshold_for_nr_of_votes = nr_of_most_votes // 2  # We stop if the number of votes drops by a factor of 2.
    if nr_of_most_votes <= 1:  # Very strong condition to filter out false matches!
        return [], 0, 0.0  # In this case, there is no match, so we just return an empty list.

    matching_points = []  # Storage the matching pairs of points.
    used_points1 = set()
    used_points2 = set()
    V = 0  # Sum of the votes of the successfully matched pairs.

    for (point1, point2), nr_of_votes in voting_list:

        # Check if the current pair is still valid.
        if nr_of_votes < threshold_for_nr_of_votes:
            break

        if point1 in used_points1 or point2 in used_points2:
            break  # Since we already have a better match at this point in our matching points list.

        if nr_of_votes == 0:
            break

        matching_points.append((point1, point2))
        used_points1.add(point1)
        used_points2.add(point2)

        V += nr_of_votes
        threshold_for_nr_of_votes = nr_of_votes // 2  # Updating the threshold.

    # Calculate the ratio of voting triangles to the total number of triangles.
    f_t = len(contributing_triangles) / total_nr_triangles_before_matching

    return matching_points, V, f_t


""" STEP 7: Computing a score for each matching point patterns. """


def compute_score(V: int, f_t: float, V_max: int, S2: float):
    """ This method computes a score based on V and f_t, and then we normalize it. The higher the score, the better."""

    if V_max == 0:
        return 0, 0.0, 'No'

    S = V * f_t
    S_rel = S / V_max

    S2 = S2 * 100  # Converting to percent.
    if S2 <= 10:
        keyword = 'No'
    elif 10 < S2 <= 40:
        keyword = 'Weak'
    elif 40 < S2 <= 70:
        keyword = 'Medium'
    else:
        keyword = 'Strong'

    return S, S_rel, keyword


""" STEP 8: Displaying the best matches to the user. """


def score_combined(S1_rel, S2, weight_S1, weight_S2):
    """ This method calculates the combined score based on S1 and S2. The higher the score, the better. """

    S_combined = S1_rel * weight_S1 + S2 * weight_S2

    return S_combined


def sort_list_of_scores(list_of_scores, weight_S1, weight_S2):
    """ This method sorts the list of scores based on its parameters. We contain the matches with S1 >= 50 in the top
    list, the others go in the bottom list. Then we sort the lists, based on the combined score. And finally we
    merge the lists back together. """

    top_list = []
    bottom_list = []

    for (S1, S1_rel, S2, keyword, name) in list_of_scores:
        if S1 >= 50:
            top_list.append((S1, S1_rel, S2, keyword, name))
        else:
            bottom_list.append((S1, S1_rel, S2, keyword, name))

    top_list.sort(key=lambda x: score_combined(x[1], x[2], weight_S1, weight_S2), reverse=True)
    bottom_list.sort(key=lambda x: score_combined(x[1], x[2], weight_S1, weight_S2), reverse=True)

    return top_list + bottom_list


def display_results(image: np.ndarray, database: dict[str, np.ndarray],
                    list_of_scores: list[tuple[int, float, float, str, str]]):
    """ This method will display the three best matches to the user.

    INPUT: list_of_scores is of the form [ (S1, S1_rel, S2, keyword, name), ... ]. """

    # Select the top three matches.
    top_matches = list_of_scores[:3]

    # Plot the unknown image and top matches with information on one plot.
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    axes[0].imshow(image)
    axes[0].set_title("Unknown Image")
    axes[0].axis('off')

    for i, (S1, S1_rel, S2, keyword, name) in enumerate(top_matches):
        S2 = "%.2f" % round(S2 * 100, 2)
        S2 = str(S2) + '%'

        image_database = database.get(name, None)
        image_database = cv.cvtColor(image_database, cv.COLOR_BGR2RGB)  # Load and convert the image data
        axes[i + 1].imshow(image_database)
        axes[i + 1].set_title(name.split('.')[0])
        axes[i + 1].axis('off')
        # Display the scores beneath the image
        axes[i + 1].text(0.5, -0.1, f"Score1: {S1:.2f}\nScore2: {S2}\n{keyword} match",
                         transform=axes[i + 1].transAxes, ha='center', va='top', fontsize=10)

    # Add a vertical black line between the unknown image and the first match
    line_x = (axes[0].get_position().x1 + axes[1].get_position().x0) / 2 - 0.055
    fig.add_artist(plt.Line2D((line_x, line_x), (0, 1), color='black', linewidth=2,
                              transform=fig.transFigure))

    plt.tight_layout()
    plt.show()
