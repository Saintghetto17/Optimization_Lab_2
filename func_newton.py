import random
import typing
import sympy as sp
from matplotlib.axes import Axes
from prettytable import PrettyTable
from scipy import optimize
import enum
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.filterwarnings("ignore")

x, y = sp.symbols('x y')


class NAME(enum.Enum):
    CONSTANT_STEP = "CONSTANT STEP NEWTON"
    CHANGING_STEP_TERNARY = "TERNARY STEP NEWTON"


class FUNCTION(enum.Enum):
    FUNC_1 = x ** 2 + (2 * x - 4 * y) ** 2 + (x - 5) ** 2
    FUNC_2 = x ** 2 + y ** 2 - x * y + 2 * x - 4 * y + 3
    FUNC_3 = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


class REGIME(enum.Enum):
    CONSTANT_STEP = 0
    CHANGING_STEP_TERNARY = 1


FUNCTIONS: list[FUNCTION] = [FUNCTION.FUNC_1, FUNCTION.FUNC_2, FUNCTION.FUNC_3]
GLOBAL_MIN: list[float] = [25 / 2, -1, 0]
TYPES_METHODS: list[NAME] = [NAME.CONSTANT_STEP, NAME.CHANGING_STEP_TERNARY]
REGIMES: list[REGIME] = [REGIME.CONSTANT_STEP, REGIME.CHANGING_STEP_TERNARY]
DISPLAY_FUNCTION = ["x^2 + (2x - 4y)^2 + (x-5)^2", "x^2 + y^2 - xy + 2x - 4y + 3", "(1 - x)^2 + 100(y - x^2)^2"]


def gradient(dot: tuple[float, float], func: FUNCTION) -> tuple[float, ...]:
    """
    Calculates gradient of function : R^2 -> R
    :param dot: Dot at which to compute the gradient
    :param func: Function for which to compute the gradient : R^2 -> R
    :return: Calculated gradient
    """
    gradient_symbolic = [sp.diff(func.value, var) for var in (x, y)]
    compute_gradient = sp.lambdify((x, y), gradient_symbolic, 'numpy')
    gradient_at_point = compute_gradient(dot[0], dot[1])
    return gradient_at_point[0], gradient_at_point[1]


EPS_SEARCH = 0.000001


def ternary_search_min(func: typing.Callable[[tuple[float, float]], float],
                       left: float,
                       right: float,
                       grad: tuple[float, ...],
                       dot: tuple[float, float]) -> float:
    """
    Ternary Search for finding the minimum on interval. F : R -> R
    :param func: Function for finding a minimum : R -> R
    :param left: Left board of search
    :param right: Right board of search
    :param grad: Gradient of the given function, calculated on the previous step of gradient descent
    :param dot: previous dot x_{k} (check docs for next_step)
    :return: Founded min value
    """
    if right - left < EPS_SEARCH:
        return (left + right) / 2
    a: float = (left * 2 + right) / 3
    b: float = (left + right * 2) / 3
    a_dot: tuple[float, float] = (dot[0] - a * grad[0], dot[1] - a * grad[1])
    b_dot: tuple[float, float] = (dot[0] - b * grad[0], dot[1] - b * grad[1])
    if func(a_dot) < func(b_dot):
        return ternary_search_min(func, left, b, grad, dot)
    else:
        return ternary_search_min(func, a, right, grad, dot)


def get_s_k_hesse(func: FUNCTION, dot: tuple[float, float], grad_vector: tuple[float, ...]) -> tuple[float, ...]:
    hesse_matrix = sp.hessian(func.value, (x, y))
    prev_point = {x: dot[0], y: dot[1]}
    hessian_at_point = np.array(hesse_matrix.subs(prev_point), dtype=float)
    inverse_hessian_at_point = np.linalg.inv(hessian_at_point)
    product_vector = np.dot(inverse_hessian_at_point, np.array([grad_vector[0], grad_vector[1]]))
    return product_vector


def next_step(prev_dot: tuple[float, float],
              gradient_vector: tuple[float, ...],
              regime: REGIME,
              func: FUNCTION,
              constant_step: float | None = None) -> tuple[float, float]:
    """
    The help function inside gradient algorithm. Used to step from x_{k} -> x_{k+1}.
    :param prev_dot: x_{k}
    :param gradient_vector: Gradient of function
    :param regime: The type of gradient that shows how to find x_{k+1}
    :param func: The function for finding minimum value
    :param constant_step: Optional. If regime is set to LEARNING_RATE, then it should not be None
    :return: x_{k+1}
    """

    s_k = get_s_k_hesse(func, prev_dot, gradient_vector)
    if regime == REGIME.CONSTANT_STEP:
        return prev_dot[0] - constant_step * s_k[0], prev_dot[1] - constant_step * s_k[1]
    elif regime == REGIME.CHANGING_STEP_TERNARY:
        step: float = ternary_search_min(lambda xy: func.value.subs({x: xy[0], y: xy[1]}), -5, 5,
                                         s_k, prev_dot)
        return prev_dot[0] - step * s_k[0], prev_dot[1] - step * s_k[1]


def normalize(dot_1: tuple[float, float], dot_2: tuple[float, float]) -> float:
    """
    Function to calculate the norm of two dots: X -> R
    :param dot_1: First dot
    :param dot_2: Second dot
    :return: the calculated real value of norm
    """
    return sqrt((dot_1[0] - dot_2[0]) ** 2 + (dot_1[1] - dot_2[1]) ** 2)


# Contract: return value : tuple[0] -> counted value; tuple[1] -> number of iterations
def newton(initial_dot: tuple[float, float],
           eps: float,
           func: FUNCTION,
           regime: REGIME,
           learning_rate: float | None = None) -> tuple[float, int, [list[float], list[float], list[float]]]:
    """
    The gradient descent algorithm for any type of gradient method
    :param initial_dot: Dot from which to start calculating gradient
    :param by_dot_normalize: The STOP criterion by dot
    :param by_func_normalize: The STOP criterion by function values
    :param eps: The deviation at which to stop calculating gradient
    :param func: The function that will be used to find her minimum
    :param regime: The regime used to calculate step of gradient
    :param learning_rate: The constant step if the regime is LEARNING RATE, otherwise None
    :return: The found min value, the number of iterations, the dots of gradient root
    """
    prev_dot: tuple[float, float] = initial_dot
    iterations: int = 0
    cord_data = [[], [], []]
    while True:
        iterations += 1
        current_gradient: tuple[float, ...] = gradient(prev_dot, func)
        current_dot: tuple[float, float] = next_step(prev_dot, current_gradient, regime, func,
                                                     constant_step=learning_rate)
        current_func_value: float = func.value.subs({x: current_dot[0], y: current_dot[1]})
        cord_data[0].append(current_dot[0])
        cord_data[1].append(current_dot[1])
        cord_data[2].append(current_func_value)
        if normalize(current_dot, prev_dot) <= eps:
            return current_func_value, iterations, cord_data
        prev_dot = current_dot


######################################################## LEARNING RATE | FAST RATE ##############################################
# Every point will be gone throw every learning rate for analysis
INIT_POINTS: list[tuple[float, float] | tuple[None, None]] = []  # start points
EPSILON = []  # EPS at which algorithm will stop
CONSTANT_STEPS = []  # different learning rates
for i in range(2):
    INIT_POINTS.append((random.randint(1, 5), random.randint(1, 5)))
    CONSTANT_STEPS.append(random.uniform(0.5, 1.5))
    EPSILON.append(random.uniform(0, 0.001))

legend_data = [[], []]
legend_data2D1 = [[], []]
legend_data2D2 = [[], []]


def fill_tables(col_names: list[str], tables: list[PrettyTable],
                datas: list[list[typing.Any]], newton_name: NAME, regime: REGIME, results: list[list[tuple]]) -> None:
    experiment_number = 0
    for func in range(len(FUNCTIONS)):
        tables.append(PrettyTable(col_names))
        datas.append([])
        for i in range(len(INIT_POINTS)):
            for j in range(len(EPSILON)):
                datas[func].append(experiment_number)
                datas[func].append(DISPLAY_FUNCTION[func])
                datas[func].append(GLOBAL_MIN[func])
                datas[func].append(INIT_POINTS[i])
                datas[func].append(results[func][j + len(INIT_POINTS) * i][1])
                datas[func].append(EPSILON[j])
                if regime == REGIME.CONSTANT_STEP:
                    datas[func].append(CONSTANT_STEPS[j])
                elif newton_name == NAME.CHANGING_STEP_TERNARY:
                    datas[func].append(NAME.CHANGING_STEP_TERNARY.value)
                datas[func].append(results[func][j + len(INIT_POINTS) * i][0])
                experiment_number += 1


def fill_graphic(ax_fig: Axes,
                 ax_fig_2D1: Axes,
                 ax_fig_2D2: Axes, results: list[list[tuple]], regime: REGIME, numbers_to_display: list[int],
                 newton_name: NAME):
    exp_cnt = 0
    for func in range(len(FUNCTIONS)):
        results.append([])
        for i in range(len(INIT_POINTS)):
            for j in range(len(EPSILON)):
                learning_rate = CONSTANT_STEPS[j]
                try:
                    buffer = newton(INIT_POINTS[i], EPSILON[j], FUNCTIONS[func],
                                    regime,
                                    learning_rate=learning_rate)
                    results[func].append(buffer[:2])
                    if exp_cnt in numbers_to_display:
                        l, = ax_fig.plot(buffer[2][0], buffer[2][1], buffer[2][2], '-')
                        ax_fig.scatter(buffer[2][0], buffer[2][1], buffer[2][2])
                        legend_data[0].append(l)
                        legend_data[1].append(newton_name.value + " " + str(exp_cnt))
                        if ax_fig_2D1 is not None and ax_fig_2D2 is not None:
                            points_x = buffer[2][0]
                            points_y = buffer[2][1]
                            l2d, = ax_fig_2D2.plot(points_x, points_y)
                            if func == 0:
                                legend_data2D1[0].append(l2d)
                                legend_data2D1[1].append(newton_name.value + " " + str(exp_cnt))
                            if func == 1:
                                legend_data2D2[0].append(l2d)
                                legend_data2D2[1].append(newton_name.value + " " + str(exp_cnt))
                except OverflowError:
                    results[func].append((None, None))
                exp_cnt += 1


def fill_data(col_names: list[str],
              tables: list[PrettyTable],
              datas: list[list[typing.Any]],
              ax_fig: Axes,
              ax_fig_2D1: Axes,
              ax_fig_2D2: Axes,
              newton_name: NAME, regime: REGIME,
              numbers_to_display: list[int]) -> None:
    """
    Fills the list of datas to put in the tables
    :param col_names: Names of the columns in the tables
    :param tables: List of tables
    :param datas: List of data to plot ordered by the column names
    :param ax_fig: Instance of the Axes figure for plotting
    :param ax_fig_2D1: Instance of the Axes figure for plotting on level_line_graph1
    :param ax_fig_2D2: Instance of the Axes figure for plotting on level_line_graph2
    :param newton_name: The type of gradient with string values
    :param regime: The regime of the gradient with integer values
    :param numbers_to_display: Number of rows in the table to display
    :return: None
    """

    # results tuple(iterations, value)
    results: list[list[tuple]] = []
    fill_graphic(ax_fig, ax_fig_2D1, ax_fig_2D2, results, regime, numbers_to_display, newton_name)
    fill_tables(col_names, tables, datas, newton_name, regime, results)


def print_tables(cols: list[str], tables: list[PrettyTable], datas: list[list[typing.Any]]):
    """
    Visualize tables with data for analysis
    :param cols: The names of the columns in the tables
    :param tables: The instances of the tables to put data
    :param datas: The data to put in the tables ordered by the column names
    :return: None
    """
    number_of_cols: int = len(cols)
    for i in range(len(datas)):
        data: list[typing.Any] = datas[i]
        while data:
            tables[i].add_row(data[:number_of_cols])
            data = data[number_of_cols:]
    for table in tables:
        print(table)
        print()


figure_method_steps = plt.figure()
ax_fms = figure_method_steps.add_subplot(projection='3d')

# VISUALIZATION OF OUR GRAPHICS

x_coords = y_coords = np.arange(-3, 3, 0.001)
X_coords, Y_coords = np.meshgrid(x_coords, y_coords)
func_1 = sp.lambdify((x, y), FUNCTION.FUNC_1.value, 'numpy')
func_2 = sp.lambdify((x, y), FUNCTION.FUNC_2.value, 'numpy')

Z1 = np.array(func_1(np.ravel(X_coords), np.ravel(Y_coords))).reshape(X_coords.shape)
Z2 = np.array(func_2(np.ravel(X_coords), np.ravel(Y_coords))).reshape(X_coords.shape)

figure = plt.figure(figsize=(14, 10))

ax1 = figure.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(X_coords, Y_coords, Z1)
ax2 = figure.add_subplot(2, 2, 2)
levels_list1 = [12.5, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]
cs1 = ax2.contour(X_coords, Y_coords, Z1, levels=levels_list1)
cs1.clabel()

ax4 = figure.add_subplot(2, 2, 4)
levels_list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cs2 = ax4.contour(X_coords, Y_coords, Z2, levels=levels_list2)

cs2.clabel()


def show_result():
    for i in range(len(TYPES_METHODS)):
        print(
            "################################################# {TYPE} ###############################################".format(
                TYPE=TYPES_METHODS[i].value))
        column_names: list[str] = ['№', 'FUNCTION', 'GLOBAL_MIN', 'INIT_POINT', 'ITERATIONS', 'EPS',
                                   TYPES_METHODS[i].value,
                                   'VALUE']
        tables: list[PrettyTable] = []
        datas: list[list[typing.Any]] = []
        fill_data(column_names, tables, datas,
                  ax_fms, ax2, ax4, TYPES_METHODS[i], REGIMES[i], [1, 10])
        print_tables(column_names, tables, datas)
    plt.show()


show_result()
