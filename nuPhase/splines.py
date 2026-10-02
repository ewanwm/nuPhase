import nuTens as nt
import numpy as np

from nuTens import dtype, units, tensor, autograd
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

    def __init__(
        self,
        parameters: typing.List[str],
        knot_x_positions: typing.List[np.ndarray],
        knot_y_positions: typing.List[np.ndarray],
        interpolation: InterpolationBase
    ):

        self.interpolation: InterpolationBase = interpolation

        self.parameter_names: typing.List[str] = parameters
        self.n_params: int = len(self.parameter_names)

        self.knot_x_positions: typing.List[np.ndarray] = knot_x_positions
        self.knot_y_positions: typing.List[np.ndarray] = knot_y_positions

        if len(knot_x_positions) != self.n_params:

            raise ValueError("Knot position shape does not match size of parameter name list!")

        if len(knot_y_positions) != self.n_params:

            raise ValueError("Knot array shape does not match size of parameter name list!")

        ## check that all arrays have same number of rows (events / bins)
        for param_knots in self.knot_y_positions[1:]:

            if param_knots.shape[0] != self.knot_y_positions[0].shape[0]:
                raise ValueError("Knot array shapes are inconsistent!")

        self.n_rows: int = self.knot_y_positions[0].shape[0]
        self.n_knots: typing.List[int] = [self.knot_x_positions[i_param].shape[0] for i_param in range(self.n_params)]

        self.segment_dx: typing.List[np.ndarray] = [np.zeros(self.n_knots[i_param] + 1) for i_param in range (self.n_params)]
        for i_param in range(self.n_params):
            self.segment_dx[i_param][1:-1] = self.knot_x_positions[i_param][1:] - self.knot_x_positions[i_param][:-1]
            self.segment_dx[i_param][0]  = self.segment_dx[i_param][1]
            self.segment_dx[i_param][-1] = self.segment_dx[i_param][-2]
                                                    
        ## Create coefficient array that will be filled using user supplied coefficient calculator
        self.coefficients_array: typing.List[np.ndarray] = []
        for i_param in range(self.n_params):

            param_n_knots = self.knot_x_positions[i_param].shape[0]
            values = np.zeros((self.n_rows, param_n_knots + 1, 3))
            self.coefficients_array.append(values)

        self.calculate_coefficients()

    def calculate_coefficients(self):

        ## calculate coefficients for each parameter
        for i_param in range(self.n_params):

            interpolation_coefficients = self.interpolation.calculate_coefficients(
                self.knot_x_positions[i_param], self.knot_y_positions[i_param]
            )

            ## check the shape of the coeficients returned by interpolation object
            if self.coefficients_array[i_param][:, 1:-1, :].shape != interpolation_coefficients.shape:
                raise ValueError(f"interpolation coefficients for parameter {self.parameter_names[i_param]} from {self.interpolation.name} are the wrong shape! Expected{self.coefficients_array[i_param][:, 1:-1, :].shape} but got {interpolation_coefficients.shape}")
            
            self.coefficients_array[i_param][:, 1:-1, :] = interpolation_coefficients
            self.coefficients_array[i_param][:, 0, :] = self.coefficients_array[i_param][:, 1, :]
            self.coefficients_array[i_param][:, -1, :] = self.coefficients_array[i_param][:, -2, :]

    def evaluate(self, parameter_values: Tensor):
        
        parameter_values_array = parameter_values.numpy()

        if parameter_values_array.shape[0] != self.n_params:
            raise ValueError(f"Wrong number of values! expected {self.n_params} but got {parameter_values_array.shape[0]}")

        ## get which segment the parameter values lie in
        segment_indices = []
        for i_param in range(self.n_params):
            
            index = np.digitize(parameter_values_array[i_param], self.knot_x_positions[i_param])

            segment_indices.append(index)

        weights = Tensor.ones((self.n_rows,))

        for i_param in range(self.n_params):

            segment_index = segment_indices[i_param]

            ## get the position of the anchor knot
            x0 = self.knot_x_positions[i_param][np.clip(segment_index - 1, 0, self.n_knots[i_param] -2)]
            y0 = self.knot_y_positions[i_param][:, np.clip(segment_index - 1, 0, self.n_knots[i_param] -2)]

            ## higher order polynomial values
            t = (parameter_values.get_values([i_param]) + -Tensor(x0))
            squared = tensor.mul(t, t)
            cubed   = tensor.mul(squared, t)

            ## TODO: Have the "current" coefficients cached and check if the segment index has changed before fetching whole new ones
            segment_coefficients_1 = self.coefficients_array[i_param][:, segment_index, 0]
            segment_coefficients_2 = self.coefficients_array[i_param][:, segment_index, 1]
            segment_coefficients_3 = self.coefficients_array[i_param][:, segment_index, 2]

            ## tensorify them
            self.coefficients_1_tensor = Tensor(segment_coefficients_1)
            self.coefficients_2_tensor = Tensor(segment_coefficients_2)
            self.coefficients_3_tensor = Tensor(segment_coefficients_3)

            ## calculate weights for the current parameter
            param_weights = (
                Tensor(y0) + 
                tensor.mul(t, self.coefficients_1_tensor) + 
                tensor.mul(squared, self.coefficients_2_tensor) + 
                tensor.mul(cubed, self.coefficients_3_tensor)
            )

            ## add to total weight
            weights = tensor.mul(weights, param_weights)

        return weights
