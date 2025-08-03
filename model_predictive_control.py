import numpy as np
from scipy.optimize import minimize
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictiveControl:
    """
    Model Predictive Control class for automated vehicles.

    Implements model predictive control (MPC) for vehicle trajectory tracking and velocity
    planning. Includes prediction of vehicle state and calculation of control inputs based on
    optimization.

    ...

    Attributes
    ----------
    vehicle_model : VehicleModel
        Model of the vehicle dynamics.
    horizon : int
        Prediction horizon for MPC.
    num_steps : int
        Number of time steps for prediction.
    dt : float
        Time step duration.
    weights : Dict[str, float]
        Weights for the cost function.
    x_ref : List[float]
        Reference trajectory for x-position.
    v_ref : List[float]
        Reference trajectory for velocity.
    u_min : float
        Minimum control input (steering angle).
    u_max : float
        Maximum control input (steering angle).
    u_delta_max : float
        Maximum change in control input.
    x_min : float
        Minimum x-position for constraints.
    x_max : float
        Maximum x-position for constraints.
    v_min : float
        Minimum velocity for constraints.
    v_max : float
        Maximum velocity for constraints.
    constraints : List[Dict]
        List of constraints for the optimization problem.

    Methods
    -------
    predict_vehicle_state(self, x0: List[float], u: List[float]) -> List[List[float]]
        Predict the vehicle state over the prediction horizon.

    calculate_control_inputs(self, x0: List[float]) -> List[float]
        Calculate the optimal control inputs for the given initial state.
    """

    def __init__(self, vehicle_model: 'VehicleModel', horizon: int, dt: float, weights: Dict[str, float],
                 x_ref: List[float], v_ref: List[float], u_min: float, u_max: float, u_delta_max: float,
                 x_min: float = None, x_max: float = None, v_min: float = 0, v_max: float = float('inf'),
                 constraints: List[Dict] = None):
        """
        Initialize the ModelPredictiveControl class.

        Parameters
        ----------
        vehicle_model : VehicleModel
            Model of the vehicle dynamics.
        horizon : int
            Prediction horizon for MPC.
        dt : float
            Time step duration.
        weights : Dict[str, float]
            Weights for the cost function. Should include 'pos', 'vel', 'input', and optionally 'lat_acc'.
        x_ref : List[float]
            Reference trajectory for x-position over the horizon.
        v_ref : List[float]
            Reference trajectory for velocity over the horizon.
        u_min : float
            Minimum control input (steering angle).
        u_max : float
            Maximum control input (steering angle).
        u_delta_max : float
            Maximum change in control input.
        x_min : float, optional
            Minimum x-position for constraints, by default None.
        x_max : float, optional
            Maximum x-position for constraints, by default None.
        v_min : float, optional
            Minimum velocity for constraints, by default 0.
        v_max : float, optional
            Maximum velocity for constraints, by default infinity.
        constraints : List[Dict], optional
            Additional constraints for the optimization problem, by default None.

        Raises
        ------
        ValueError
            If weights do not contain required keys or if horizon is less than 2.
        """
        if horizon < 2:
            raise ValueError("Horizon must be at least 2.")
        if any(weight not in weights for weight in ['pos', 'vel', 'input']):
            raise ValueError("Weights must include 'pos', 'vel', and 'input'.")

        self.vehicle_model = vehicle_model
        self.horizon = horizon
        self.num_steps = horizon + 1
        self.dt = dt
        self.weights = weights
        self.x_ref = x_ref
        self.v_ref = v_ref
        self.u_min = u_min
        self.u_max = u_max
        self.u_delta_max = u_delta_max
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max
        self.constraints = constraints if constraints is not None else []

        # Additional attributes
        self.x_prev = None
        self.u_prev = None

    def predict_vehicle_state(self, x0: List[float], u: List[float]) -> List[List[float]]:
        """
        Predict the vehicle state over the prediction horizon.

        Uses the vehicle model and provided control inputs to predict the vehicle state at each
        time step.

        Parameters
        ----------
        x0 : List[float]
            Initial state of the vehicle: [x_pos, y_pos, velocity, yaw_angle, yaw_rate].
        u : List[float]
            Control inputs (steering angles) for each time step.

        Returns
        -------
        List[List[float]]
            Predicted vehicle state at each time step: [[x0, y0, v0, psi0, delta0], ..., [xn, yn, vn, psin, deltan]].

        Raises
        ------
        ValueError
            If the length of x0 or u is incorrect.
        """
        if len(x0) != 5:
            raise ValueError("Initial state x0 must have length 5.")
        if len(u) != self.num_steps - 1:
            raise ValueError("Control inputs u must have length num_steps - 1.")

        predicted_state = [x0]
        for i in range(self.num_steps - 1):
            x_next = self.vehicle_model.update_state(x0=x0, u=u[i], dt=self.dt)
            predicted_state.append(x_next)
            x0 = x_next

        return predicted_state

    def calculate_cost(self, x_pred: List[List[float]], u: List[float]) -> float:
        """
        Calculate the cost for the predicted vehicle state and control inputs.

        Uses the reference trajectories and weights to compute the cost.

        Parameters
        ----------
        x_pred : List[List[float]]
            Predicted vehicle state at each time step.
        u : List[float]
            Control inputs used for the prediction.

        Returns
        -------
        float
            Total cost for the predicted state and control inputs.
        """
        cost = 0
        for i in range(self.num_steps):
            pos_error = x_pred[i][0] - self.x_ref[i]
            vel_error = x_pred[i][2] - self.v_ref[i]

            cost += self.weights['pos'] * pos_error**2 + self.weights['vel'] * vel_error**2
            if 'input' in self.weights:
                cost += self.weights['input'] * u[i]**2
            if 'lat_acc' in self.weights and i > 0:
                lat_acc = (x_pred[i][4] - x_pred[i-1][4]) / self.dt
                cost += self.weights['lat_acc'] * lat_acc**2

        return cost

    def calculate_control_inputs(self, x0: List[float]) -> List[float]:
        """
        Calculate the optimal control inputs for the given initial state.

        Uses nonlinear optimization to find the control inputs that minimize the cost function
        over the prediction horizon.

        Parameters
        ----------
        x0 : List[float]
            Initial state of the vehicle: [x_pos, y_pos, velocity, yaw_angle, yaw_rate].

        Returns
        -------
        List[float]
            Optimal control inputs for each time step.

        Raises
        ------
        ValueError
            If the optimization fails to converge.
        """
        # Initial control inputs (steering angles)
        u0 = [self.u_prev] if self.u_prev is not None else [0]
        u0 += [0] * (self.num_steps - len(u0) - 1)

        # Predict initial state with previous control input
        x_init = self.predict_vehicle_state(x0, u0)

        # Define bounds and constraints for optimization
        bounds = self.num_steps * [(self.u_min, self.u_max)]
        constraints = self.constraints + [{"type": "eq", "fun": lambda u: np.diff(u) - self.u_delta_max}]

        # Optimize control inputs
        options = {"maxiter": 100}
        result = minimize(self.calculate_cost, u0, args=(x_init,), bounds=bounds, constraints=constraints, options=options)

        if not result.success:
            raise ValueError("Optimization failed to converge. Initial state:", x0)

        # Extract optimal control inputs
        u_opt = result.x

        # Store previous state and control input for next iteration
        self.x_prev = x0
        self.u_prev = u_opt[-1]

        return u_opt

class VehicleModel:
    """
    Vehicle dynamics model for use in Model Predictive Control.

    Implements a kinematic bicycle model to predict the vehicle state based on control inputs.

    ...

    Attributes
    ----------
    L : float
        Distance between front and rear axle (wheelbase).
    lf : float
        Distance from center of gravity to front axle.

    Methods
    -------
    update_state(self, x0: List[float], u: float, dt: float) -> List[float]
        Update the vehicle state based on control input and time step.
    """

    def __init__(self, L: float, lf: float):
        """
        Initialize the VehicleModel class.

        Parameters
        ----------
        L : float
            Distance between front and rear axle (wheelbase).
        lf : float
            Distance from center of gravity to front axle.
        """
        self.L = L
        self.lf = lf

    def update_state(self, x0: List[float], u: float, dt: float) -> List[float]:
        """
        Update the vehicle state based on control input and time step.

        Uses the kinematic bicycle model to predict the next state of the vehicle.

        Parameters
        ----------
        x0 : List[float]
            Current state of the vehicle: [x_pos, y_pos, velocity, yaw_angle, yaw_rate].
        u : float
            Control input (steering angle).
        dt : float
            Time step duration.

        Returns
        -------
        List[float]
            Next state of the vehicle: [x_pos, y_pos, velocity, yaw_angle, yaw_rate].
        """
        x, y, v, psi, delta = x0

        # Calculate next state using kinematic bicycle model
        v_f = v * np.cos(delta)
        v_r = v * np.sin(delta)
        x_f = x + v_f * dt * np.cos(psi)
        y_f = y + v_f * dt * np.sin(psi)
        psi_f = psi + v_f * dt / self.lf * np.sin(delta)
        v_f_f = v_f * (1 - v_f * dt / self.L * np.sin(delta))

        x_r = x - v_r * dt * np.cos(psi + np.pi / 2)
        y_r = y - v_r * dt * np.sin(psi + np.pi / 2)
        psi_r = psi + v_r * dt / self.lf * np.sin(delta)
        v_r_f = v_r * (1 + v_r * dt / self.L * np.sin(delta))

        x_next = x_f - (x_f - x_r) / 2
        y_next = y_f + (y_f - y_r) / 2
        psi_next = psi_f + (psi_f - psi_r) / 2
        v_next = (v_f_f + v_r_f) / 2
        delta_next = delta

        return [x_next, y_next, v_next, psi_next, delta_next]

# Example usage
if __name__ == "__main__":
    # Vehicle model parameters
    L = 2.9  # wheelbase
    lf = 1.11  # distance from CG to front axle

    # MPC parameters
    horizon = 10
    dt = 0.1
    weights = {'pos': 1, 'vel': 1, 'input': 1, 'lat_acc': 1}
    x_ref = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # example reference trajectory
    v_ref = [5] * horizon  # constant velocity reference
    u_min = -np.pi / 4
    u_max = np.pi / 4
    u_delta_max = np.pi / 8

    # Initial state
    x0 = [0, 0, 5, 0, 0]  # initial position, velocity, yaw angle, yaw rate

    # Create vehicle model and MPC object
    vehicle_model = VehicleModel(L, lf)
    mpc = ModelPredictiveControl(vehicle_model, horizon, dt, weights, x_ref, v_ref, u_min, u_max, u_delta_max)

    # Calculate optimal control inputs
    optimal_u = mpc.calculate_control_inputs(x0)

    # Predict vehicle state with optimal control inputs
    predicted_state = mpc.predict_vehicle_state(x0, optimal_u)

    # Print results
    logger.info("Optimal control inputs: %s", optimal_u)
    logger.info("Predicted vehicle state: %s", predicted_state)