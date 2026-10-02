import numpy as np

from nuTens.tensor import Tensor

import math as m
import typing
import abc

class InterpolationBase(abc.ABC):
    """Base class for all interpolation functions
    """

    def __init__(self, name: str):

        self.name = name

    @abc.abstractmethod
    def calculate_coefficients(self, knot_postions: np.ndarray, knot_array: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError()

class MonotonicInterpolation(InterpolationBase):
    """Cubic interpolation with monotonicity (between knots) enforced
    """

    def __init__(self):

        super().__init__("monotonic_cubic")

    def calculate_coefficients(self, knot_x_postions: np.ndarray, knot_y_positions: np.ndarray) -> np.ndarray:

        dx = knot_x_postions[1:] - knot_x_postions[:-1]
        dy = knot_y_positions[:, 1:] - knot_y_positions[:, :-1]

        n_rows = knot_y_positions.shape[0]
        n_knots = knot_x_postions.shape[0]

        secants = (knot_y_positions[:, 1:] - knot_y_positions[:, :-1]) / dx

        tangents = np.zeros(knot_y_positions.shape)
        tangents[:, 0]  = secants[:, 0]
        tangents[:, -1] = secants[:, -1]

        tangents[:, 1:-1] = (secants[:, 1:] + secants[:, :-1]) / 2.0

        ## now rescale tangents to enforce monotonicity
        for i_row in range(n_rows):
            for i_knot in range(n_knots - 1):

                if secants[i_row, i_knot] == 0.0:
                    tangents[i_row, i_knot] = 0.0
                    tangents[i_row, i_knot + 1] = 0.0

                    continue

                alpha = tangents[i_row, i_knot]   / secants[i_row, i_knot]
                beta  = tangents[i_row, i_knot+1] / secants[i_row, i_knot]

                if alpha < 0.0:
                    tangents[i_row, i_knot] = 0.0

                if beta < 0.0:
                    tangents[i_row, i_knot+1] = 0.0

                if alpha * alpha + beta * beta >9.0: 
                    tau = 3.0 / m.sqrt(alpha * alpha + beta * beta)
                    tangents[i_row, i_knot] = tau * alpha * secants[i_row, i_knot];
                    tangents[i_row, i_knot +1 ] = tau * beta  * secants[i_row, i_knot];

        ## now calculate coefficients

        b = tangents[:, :-1] * dx
        c = 3.0 * dy -2.0 * dx * tangents[:, :-1] - dx * tangents[:, 1:]
        d = - 2.0 * dy + dx * (tangents[:, 1:] + tangents[:, :-1])

        coefficients = np.zeros((n_rows, n_knots - 1, 3))

        coefficients[..., 0] = b / dx
        coefficients[..., 1] = c / (dx * dx)
        coefficients[..., 2] = d / (dx * dx * dx)

        return coefficients
    
class LinearInterpolation(InterpolationBase):
    """Piecewise linear interpolation
    """

    def __init__(self):

        super().__init__("linear")

    def calculate_coefficients(self, knot_x_postions: np.ndarray, knot_y_positions: np.ndarray) -> np.ndarray:

        dx = knot_x_postions[1:] - knot_x_postions[:-1]
        dy = knot_y_positions[:, 1:] - knot_y_positions[:, :-1]

        n_rows = knot_y_positions.shape[0]
        n_knots = knot_x_postions.shape[0]

        coefficients = np.zeros((n_rows, n_knots - 1, 3))

        coefficients[..., 0] = dy / dx

        return coefficients

class SplineArray:
    """An array of splines for some parameter
    """

    def __init__(
        self,
        knot_x_positions: np.ndarray,
        knot_y_positions: np.ndarray,
        interpolation: InterpolationBase
    ):
        """
        :param knot_x_positions: X positions of knots for the parameter - should be a 1D array (n_knots)
        :type knot_x_positions: np.ndarray
        :param knot_y_positions: Y positions of knots for the parameter - should be 2D array with shape (n_rows, n_knots)
        :type knot_y_positions: np.ndarray
        :param interpolation: The interpolation function to use to calculate segment coefficients
        :type interpolation: InterpolationBase
        :raises ValueError: If the provided arrays have the wrong number of dimensions or if the number of knots implied by the knot_x_positions array and knot_y_positions array are inconsistent
        :raises TypeError: If the provided knot_x_positions or knot_y_positions are not numpy arrays
        """

        self.interpolation: InterpolationBase = interpolation

        self.knot_x_positions: np.ndarray = knot_x_positions
        self.knot_y_positions: np.ndarray = knot_y_positions

        if type(knot_x_positions) != np.ndarray:
            raise TypeError(f"Knot X position array must be numpy array (is {type(knot_x_positions)})")
        if type(knot_y_positions) != np.ndarray:
            raise TypeError(f"Knot Y position array must be numpy array (is {type(knot_y_positions)})")
        if knot_x_positions.ndim != 1:
            raise ValueError(f"Knot X position array must be 1 dimensional! got {knot_x_positions.ndim}")
        if knot_y_positions.ndim != 2:
            raise ValueError(f"Knot Y position array must be 2 dimensional! got {knot_y_positions.ndim}")
        if knot_x_positions.shape[0] != knot_y_positions.shape[1]:
            raise ValueError(f"Shape for knot X and Y arrays is inconsistent! (x array implies {knot_y_positions.shape[0]} knots, y array implies {knot_y_positions.shape[1]})")

        ## check that all arrays have same number of rows (events / bins)
        for param_knots in self.knot_y_positions[1:]:

            if param_knots.shape[0] != self.knot_y_positions[0].shape[0]:
                raise ValueError("Knot array shapes are inconsistent!")

        self.n_rows: int = self.knot_y_positions.shape[0]
        self.n_knots: int = self.knot_x_positions.shape[0]

        ## size of segments
        self.segment_dx: typing.List[np.ndarray] = np.zeros(self.n_knots + 1)
        self.segment_dx[1:-1] = self.knot_x_positions[1:] - self.knot_x_positions[:-1]
        self.segment_dx[0]  = self.segment_dx[1]
        self.segment_dx[-1] = self.segment_dx[-2]
                                                    
        ## Create coefficient array that will be filled using user supplied coefficient calculator
        self.coefficients_array: np.ndarray = np.zeros((self.n_rows, self.n_knots + 1, 3))

        self._calculate_coefficients()

    def _calculate_coefficients(self):

        ## calculate coefficients
        interpolation_coefficients = self.interpolation.calculate_coefficients(
            self.knot_x_positions, self.knot_y_positions
        )

        ## check the shape of the coeficients returned by interpolation object
        if self.coefficients_array[:, 1:-1, :].shape != interpolation_coefficients.shape:
            raise ValueError(f"interpolation coefficients from {self.interpolation.name} are the wrong shape! Expected{self.coefficients_array[:, 1:-1, :].shape} but got {interpolation_coefficients.shape}")
        
        self.coefficients_array[:, 1:-1, :] = interpolation_coefficients

        ## TODO: Behaviour outside of interpolation bounds should be configurable!!
        self.coefficients_array[:, 0, :] = self.coefficients_array[:, 1, :]
        self.coefficients_array[:, -1, :] = self.coefficients_array[:, -2, :]

    def evaluate(self, parameter_value: Tensor) -> Tensor:
        """Evaluate the weights for a given parameter value

        :param parameter_value: The value of the parameter
        :type parameter_value: Tensor
        :raises TypeError: If the provided parameter value is not a Tensor
        :raises ValueError: If the provided value Tensor has the wrong shape
        :return: 1D Tensor of weights - shape is (n_rows) which is determined by the number of rows in the provided knot array
        :rtype: Tensor
        """

        if type(parameter_value) != Tensor:
            raise TypeError(f"parameter value should be a Tensor! (is {type(parameter_value)})")
        if parameter_value.get_shape() != []:
            raise ValueError(f"Provided parameter value should be a single scalar value!! has shape {parameter_value.get_shape()}")
        
        parameter_value_float = parameter_value.numpy()

        ## get which segment the parameter value lies in
        segment_index = np.digitize(parameter_value_float, self.knot_x_positions)

        ## get the position of the anchor knot
        x0 = self.knot_x_positions[np.clip(segment_index - 1, 0, self.n_knots -2)]
        y0 = self.knot_y_positions[:, np.clip(segment_index - 1, 0, self.n_knots -2)]

        ## get the normalised segment coordinate
        t = parameter_value - Tensor(x0)
        squared = t * t
        cubed   = squared * t

        ## TODO: Have the "current" coefficients cached and check if the segment index has changed before fetching whole new ones
        segment_coefficients_1 = self.coefficients_array[:, segment_index, 0]
        segment_coefficients_2 = self.coefficients_array[:, segment_index, 1]
        segment_coefficients_3 = self.coefficients_array[:, segment_index, 2]

        ## tensorify them
        self.coefficients_1_tensor = Tensor(segment_coefficients_1)
        self.coefficients_2_tensor = Tensor(segment_coefficients_2)
        self.coefficients_3_tensor = Tensor(segment_coefficients_3)

        ## calculate weights for the current parameter
        param_weights = (
            Tensor(y0) + 
            t       * self.coefficients_1_tensor + 
            squared * self.coefficients_2_tensor + 
            cubed   * self.coefficients_3_tensor
        )

        return param_weights
