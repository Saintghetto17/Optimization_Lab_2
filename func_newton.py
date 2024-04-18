import time
import typing
import sympy as sp
from prettytable import PrettyTable
from scipy import optimize
import enum
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as ln
import warnings

warnings.filterwarnings("ignore")

x, y = sp.symbols('x y')


class NAME(enum.Enum):
    CONSTANT_STEP = "CONSTANT STEP NEWTON"
    CHANGING_STEP_TERNARY = "TERNARY STEP NEWTON"
    NEWTON_CG = "NEWTON CG METHOD"
    QUASI_NEWTON = "QUASI-NEWTON METHOD"
    QUASI_SCIPY = "QUASI-SCIPY"
    WOLFE_CONDITION = "WOLFE CONDITION"


class FUNCTION(enum.Enum):
    FUNC_1 = x ** 2 + (2 * x - 4 * y) ** 2 + (x - 5) ** 2
    FUNC_2 = x ** 2 + y ** 2 - x * y + 2 * x - 4 * y + 3
    FUNC_3 = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    FUNC_4 = x ** 4 + y ** 4 - 4 * x * y


class REGIME(enum.Enum):
    CONSTANT_STEP = 0
    CHANGING_STEP_TERNARY = 1
    NEWTON_CG = 2
    QUASI_NEWTON = 3
    QUASI_SCIPY = 4
    WOLFE_CONDITION = 5


FUNCTIONS: list[FUNCTION] = [FUNCTION.FUNC_1, FUNCTION.FUNC_2, FUNCTION.FUNC_3, FUNCTION.FUNC_4]
GLOBAL_MIN: list[float] = [25 / 2, -1, 0, -2]
TYPES_METHODS: list[NAME] = [NAME.CONSTANT_STEP, NAME.CHANGING_STEP_TERNARY, NAME.NEWTON_CG, NAME.QUASI_NEWTON,
                             NAME.QUASI_SCIPY,
                             NAME.WOLFE_CONDITION]
REGIMES: list[REGIME] = [REGIME.CONSTANT_STEP, REGIME.CHANGING_STEP_TERNARY, REGIME.NEWTON_CG, REGIME.QUASI_NEWTON,
                         NAME.QUASI_SCIPY,
                         REGIME.WOLFE_CONDITION]
DISPLAY_FUNCTION = ["x^2 + (2x - 4y)^2 + (x-5)^2", "x^2 + y^2 - xy + 2x - 4y + 3", "(1 - x)^2 + 100(y - x^2)^2",
                    "x^4 + y^4 - 4xy"]


def gradient(dot, func: FUNCTION) -> tuple[float, ...]:
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
    """
    Uses sympy.hessian for counting hesse_matrix, numpy for matrix inversion
    :param func:
    :param dot:
    :param grad_vector:
    :return:
    """
    hesse_matrix = sp.hessian(func.value, (x, y))
    prev_point = {x: dot[0], y: dot[1]}
    hessian_at_point = np.array(hesse_matrix.subs(prev_point), dtype=float)
    inverse_hessian_at_point = np.linalg.inv(hessian_at_point)
    product_vector = np.dot(inverse_hessian_at_point, np.array([grad_vector[0], grad_vector[1]]))
    return product_vector


def step_by_wolfe_condition(func: FUNCTION, x_k: tuple[float, float], p_k, c1: float = 0.0001,
                            c2: float = 0.9) -> float:
    """
    This function generates a step size by wolfe condition :
    x_{k+1} = x_{k} + left * p_{k}
    Search left(step), while
    f(x_{k} + left * p_k) > f(x_{k}) + c_1 * left * grad(f_k)^T * p_k or
    grad(f(x_{k} + left * p_k))^T * p_k < c_2 * grad(f_k)^T * p_k
    :param func:
    :param x_k:
    :param p_k:
    :param c1: default c_1
    :param c2: default c_2
    :return:
    """
    # f(x_k + k*p_k)
    f_x_k_k_p_k = lambda k: func.value.subs({x: x_k[0] + k * p_k[0], y: x_k[1] + k * p_k[1]})
    # f(x_k)
    f_x_k = func.value.subs({x: x_k[0], y: x_k[1]})
    # grad_f(x_k)^T
    grad_vector_x_k_transposed = np.array([gradient(x_k, func)])
    # p_k
    p_k_vector = np.array([[pk] for pk in p_k])
    # grad_f(x_k)^T * p_k
    grad_f_k_transposed_p_k_prod = (grad_vector_x_k_transposed @ p_k_vector)[0][0]
    left = -5
    right = 5
    while left < right:
        # grad_f(x_k + k * p_k) ^ T
        grad_vector_x_k_k_p_k_transposed = np.array([gradient((x_k[0] + left * p_k[0], x_k[1] + left * p_k[1]), func)])
        # grad_f(x_k + k * p_k) ^ T * p_k
        grad_vector_x_k_k_p_k_transposed_prod = (grad_vector_x_k_k_p_k_transposed @ p_k_vector)[0][0]
        if (f_x_k_k_p_k(left) <= f_x_k + c1 * left * grad_f_k_transposed_p_k_prod and
                grad_vector_x_k_k_p_k_transposed_prod >= c2 * grad_f_k_transposed_p_k_prod):
            return left
        left += 0.2
    return 1


def bfgs_method(f: FUNCTION, x_0: tuple[float, float], eps=0.001) -> tuple[float, int]:
    """
    BFGS way to create H_{k+1} matrix for Quasi-Newton's method'
    :param f: function for minimization
    :param grad: gradient function for minimization
    :param x_0: Initial guess
    :param eps: Stop
    :return:  min value of func
    """
    temp_dot_grad = gradient(x_0, f)
    func_symbolic = sp.lambdify((x, y), f.value, 'numpy')
    func_real = lambda dot: func_symbolic(dot[0], dot[1])
    k = 0
    grad_f_k = np.array([temp_dot_grad[0], temp_dot_grad[1]])
    N = len(x_0)
    I = np.eye(N, dtype=float)
    H_k = I
    x_k = np.array([x_0[0], x_0[1]])
    iterations = 0
    # grad_f_k x_k p_k
    while ln.norm(grad_f_k) > eps:
        p_k = -np.dot(H_k, grad_f_k)
        line_search = optimize.line_search(func_real, lambda dot: np.array(
            [gradient(dot, f)[0], gradient(dot, f)[1]]), x_k, p_k)
        alpha_k = line_search[0]
        x_k_1 = x_k + alpha_k * p_k
        s_k = x_k_1 - x_k
        # update x_k
        x_k = x_k_1

        temp_dot_grad_k_1 = gradient(x_k_1, f)
        grad_f_k_1 = np.array([temp_dot_grad_k_1[0], temp_dot_grad_k_1[1]])
        y_k = grad_f_k_1 - grad_f_k
        # update grad_f_k
        grad_f_k = grad_f_k_1

        iterations += 1
        r_k = 1.0 / (np.dot(y_k, s_k))
        left_matrix = I - r_k * s_k[:, np.newaxis] * y_k[np.newaxis, :]
        right_matrix = I - r_k * y_k[:, np.newaxis] * s_k[np.newaxis, :]
        H_k = np.dot(left_matrix, np.dot(H_k, right_matrix)) + (r_k * s_k[:, np.newaxis] * s_k[np.newaxis, :])
    return func_real(x_k), iterations


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
    elif regime == REGIME.WOLFE_CONDITION:
        step: float = step_by_wolfe_condition(func, prev_dot, (-s_k[0], -s_k[1]))
        return prev_dot[0] - step * s_k[0], prev_dot[1] - step * s_k[1]


def normalize(dot_1: tuple[float, float], dot_2: tuple[float, float]) -> float:
    """
    Function to calculate the norm of two dots: X -> R
    :param dot_1: First dot
    :param dot_2: Second dot
    :return: the calculated real value of norm
    """
    return sqrt((dot_1[0] - dot_2[0]) ** 2 + (dot_1[1] - dot_2[1]) ** 2)


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
    cord_data[0].append(initial_dot[0])
    cord_data[1].append(initial_dot[1])
    cord_data[2].append(func.value.subs({x: initial_dot[0], y: initial_dot[1]}))
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


# Every point will be gone throw every learning rate for analysis
INIT_POINTS: list[tuple[float, float] | tuple[None, None]] = [(1, 5), (3, 2), (0, 0)]  # start points
EPSILON = [0.001, 0.0001, 0.001]  # EPS at which algorithm will stop
CONSTANT_STEPS = [0.1, 0.01, 0.01]  # different learning rates


def fill_tables(col_names: list[str], tables: list[PrettyTable],
                datas: list[list[typing.Any]], newton_name: NAME, regime: REGIME, results: list[list[tuple]]) -> None:
    """
    This function is used to fill tables that are represented with parameters
    :param col_names:
    :param tables:
    :param datas:
    :param newton_name:
    :param regime:
    :param results:
    :return:
    """
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
                if newton_name != NAME.NEWTON_CG and newton_name != NAME.QUASI_SCIPY:
                    datas[func].append(results[func][j + len(INIT_POINTS) * i][1])
                    datas[func].append(EPSILON[j])
                    if regime == REGIME.CONSTANT_STEP:
                        datas[func].append(CONSTANT_STEPS[j])
                    else:
                        datas[func].append(newton_name.value)
                    datas[func].append(results[func][j + len(INIT_POINTS) * i][0])
                    datas[func].append(results[func][j + len(INIT_POINTS) * i][2])
                else:
                    gradient_symbolic = [sp.diff(FUNCTIONS[func].value, var) for var in (x, y)]
                    compute_gradient = sp.lambdify((x, y), gradient_symbolic, 'numpy')
                    compute_func = sp.lambdify((x, y), FUNCTIONS[func].value, 'numpy')
                    start_time = time.time()
                    if newton_name == NAME.NEWTON_CG:
                        res_optimize_scipy = optimize.minimize(lambda xy: compute_func(xy[0], xy[1]),
                                                               np.array([INIT_POINTS[i][0], INIT_POINTS[i][1]]),
                                                               method="Newton-CG",
                                                               jac=lambda xy: compute_gradient(xy[0], xy[1]))
                    else:
                        res_optimize_scipy = optimize.minimize(lambda xy: compute_func(xy[0], xy[1]),
                                                               np.array([INIT_POINTS[i][0], INIT_POINTS[i][1]]),
                                                               method="bfgs",
                                                               jac="3-point")
                    end_time = time.time()
                    datas[func].append(res_optimize_scipy.nit)
                    datas[func].append(EPSILON[j])
                    datas[func].append(newton_name.value)
                    datas[func].append(res_optimize_scipy.fun)
                    datas[func].append(end_time - start_time)
                experiment_number += 1


data_visual = [[[], []], [[], []], [[], []]]


def fill_methods_results(results: list[list[tuple]], regime: REGIME,
                         newton_name: NAME):
    """
    Fills data of steps of newton method
    :param results:
    :param regime:
    :param newton_name:
    :return:
    """
    exp_cnt = 0
    for func in range(len(FUNCTIONS)):
        results.append([])
        for i in range(len(INIT_POINTS)):
            for j in range(len(EPSILON)):
                learning_rate = CONSTANT_STEPS[j]
                start_time = time.time()
                buffer = newton(INIT_POINTS[i], EPSILON[j], FUNCTIONS[func],
                                regime,
                                learning_rate=learning_rate) if newton_name != NAME.QUASI_NEWTON else bfgs_method(
                    FUNCTIONS[func], INIT_POINTS[i], EPSILON[j])
                end_time = time.time()
                res_iter_time = buffer[:2][0], buffer[:2][1], end_time - start_time
                results[func].append(res_iter_time)
                if func != 3 and i != 2 and j != 2:
                    if newton_name != NAME.WOLFE_CONDITION and newton_name != NAME.QUASI_NEWTON:
                        if regime == REGIME.CONSTANT_STEP:
                            data_visual[func][0].append(buffer[2])
                        if regime == REGIME.CHANGING_STEP_TERNARY:
                            data_visual[func][1].append(buffer[2])
            exp_cnt += 1


def fill_data(col_names: list[str],
              tables: list[PrettyTable],
              datas: list[list[typing.Any]],
              newton_name: NAME, regime: REGIME) -> None:
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
    if newton_name != NAME.NEWTON_CG and newton_name != NAME.QUASI_SCIPY:
        fill_methods_results(results, regime, newton_name)
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


def show_result():
    """
    Shows all graphics and tables of results
    :return: void
    """
    for i in range(len(TYPES_METHODS)):
        print(
            "################################################# {TYPE} ###############################################".format(
                TYPE=TYPES_METHODS[i].value))
        column_names: list[str] = ['№', 'FUNCTION', 'GLOBAL_MIN', 'INIT_POINT', 'ITERATIONS', 'EPS',
                                   TYPES_METHODS[i].value,
                                   'VALUE', 'TIME']
        tables: list[PrettyTable] = []
        datas: list[list[typing.Any]] = []
        fill_data(column_names, tables, datas, TYPES_METHODS[i], REGIMES[i])
        print_tables(column_names, tables, datas)
    plt.show()


show_result()


def visual():
    fig = plt.figure(figsize=(21, 7))
    fig_contur = plt.figure(figsize=(21, 7))

    def calculate_z1(X, Y):
        return X ** 2 + (2 * X - 4 * Y) ** 2 + (X - 5) ** 2

    def calculate_z2(X, Y):
        return X ** 2 + Y ** 2 - X * Y + 2 * X - 4 * Y + 3

    def calculate_z3(X, Y):
        return (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(232, projection='3d')
    ax3 = fig.add_subplot(233, projection='3d')
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    ax1_contur = fig_contur.add_subplot(231)
    ax2_contur = fig_contur.add_subplot(232)
    ax3_contur = fig_contur.add_subplot(233)
    ax4_contur = fig_contur.add_subplot(234)
    ax5_contur = fig_contur.add_subplot(235)
    ax6_contur = fig_contur.add_subplot(236)

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    data_color = ["cyan", "magenta", "yellow", "red"]
    levels = [
        np.arange(0, 300, 1),
        np.arange(-1, 10, 0.1),
        np.arange(0, 4500, 10),
    ]
    x_min_point1, y_min_point1, z_min_point1 = 2.5, 1.25, 12.5
    x_min_point2, y_min_point2, z_min_point2 = 0, 2, -1
    x_min_point3, y_min_point3, z_min_point3 = 1, 1, 0

    def plotter(ax, title, x_real_min, y_real_min, z_real_min, func_index, data_visual_index, calculate_z_func,
                ax_contur, level):
        # mark real_min dot on function graphic
        ax.scatter(x_real_min, y_real_min, z_real_min, color='black', s=100)
        cnt = 0
        x_max = -1000000000
        x_min = 1000000000
        y_max = -1000000000
        y_min = 1000000000
        # plot all the steps of the newton method
        for data_set in data_visual[func_index][data_visual_index]:
            ax.plot(data_set[0], data_set[1], data_set[2], color=data_color[cnt])
            ax.scatter(data_set[0], data_set[1], data_set[2], color=data_color[cnt], s=10)
            x_max = max(x_max, max(data_set[0]))
            y_max = max(y_max, max(data_set[1]))
            x_min = min(x_min, min(data_set[0]))
            y_min = min(y_min, min(data_set[1]))
            cnt += 1
        # build grid for function and contour
        X = np.linspace(x_min, x_max, 100)
        Y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(X, Y)
        X_contur = np.linspace(x_min - 1, x_max + 1, 1000)
        Y_contur = np.linspace(y_min - 1, y_max + 1, 1000)
        X_contur, Y_contur = np.meshgrid(X_contur, Y_contur)
        Z = calculate_z_func(X, Y)
        Z_contur = calculate_z_func(X_contur, Y_contur)
        # plot the contour for graphic
        cp = ax_contur.contour(X_contur, Y_contur, Z_contur, levels=level)
        plt.colorbar(cp)
        cnt = 0
        # plot newton steps on contour
        for data_set in data_visual[func_index][data_visual_index]:
            ax_contur.plot(data_set[0], data_set[1], color=data_color[cnt], marker='o', markersize=1)
            cnt += 1
        # plot minimum dot on contour
        ax_contur.scatter(x_real_min, y_real_min, color="black", s=100)
        # plot function
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)
        ax.set_title(title)

    plotter(ax1, 'x^2 + (2x - 4y)^2 + (x-5)^2 | CONST', x_min_point1, y_min_point1, z_min_point1, 0, 0,
            calculate_z1, ax1_contur, levels[0])
    plotter(ax2, 'x^2 + y^2 - xy + 2x - 4y + 3 | CONST', x_min_point2, y_min_point2, z_min_point2, 1, 0,
            calculate_z2, ax2_contur, levels[1])
    plotter(ax3, '(1 - x)^2 + 100(y - x^2)^2  | CONST', x_min_point3, y_min_point3, z_min_point3, 2, 0,
            calculate_z3, ax3_contur, levels[2])
    plotter(ax4, 'x^2 + (2x - 4y)^2 + (x-5)^2 | TERN', x_min_point1, y_min_point1, z_min_point1, 0, 1,
            calculate_z1, ax4_contur, levels[0])
    plotter(ax5, 'x^2 + y^2 - xy + 2x - 4y + 3 | TERN', x_min_point2, y_min_point2, z_min_point2, 1, 1,
            calculate_z2, ax5_contur, levels[1])
    plotter(ax6, '(1 - x)^2 + 100(y - x^2)^2  | TERN', x_min_point3, y_min_point3, z_min_point3, 2, 1,
            calculate_z3, ax6_contur, levels[2])

    for ax in axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.show()


visual()
