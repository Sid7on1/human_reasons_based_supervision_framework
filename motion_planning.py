import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

import logging
from typing import List, Tuple, Callable

logger = logging.getLogger(__name__)

class GlobalMotionPlanner:
    """
    Global Motion Planner for automated vehicles.

    This class implements the global motion planning component of an automated vehicle system.
    It generates reference trajectories and finds optimal paths while considering dynamic constraints
    and obstacles in the environment.

    ...

    Attributes
    ----------
    config : dict
        Configuration settings for the motion planner.

    vehicle_model : VehicleModel
        Model of the automated vehicle, defining its dynamics and constraints.

    obstacles : List[np.ndarray]
        List of obstacle positions in the environment.

    tree : KDTree
        KD-tree data structure for efficient obstacle lookup.

    Methods
    -------
    generate_reference_trajectory(start, end, num_waypoints)
        Generate a reference trajectory between two points with a specified number of waypoints.

    find_optimal_path(start, end, obstacles)
        Find the optimal path between two points while avoiding dynamic obstacles.

    """

    def __init__(self, config: dict, vehicle_model: "VehicleModel"):
        """
        Initialize the GlobalMotionPlanner.

        Parameters
        ----------
        config : dict
            Configuration settings for the motion planner.

        vehicle_model : VehicleModel
            Model of the automated vehicle, defining its dynamics and constraints.

        """
        self.config = config
        self.vehicle_model = vehicle_model
        self.obstacles = []
        self.tree = None

    def generate_reference_trajectory(
        self, start: np.ndarray, end: np.ndarray, num_waypoints: int
    ) -> np.ndarray:
        """
        Generate a reference trajectory between two points with a specified number of waypoints.

        This method uses a simple interpolation technique to generate a smooth trajectory.

        Parameters
        ----------
        start : np.ndarray
            Starting position (x, y) of the trajectory.

        end : np.ndarray
            Ending position (x, y) of the trajectory.

        num_waypoints : int
            Number of waypoints in the trajectory.

        Returns
        -------
        np.ndarray
            Array of waypoints representing the reference trajectory.

        Raises
        ------
        ValueError
            If the number of waypoints is less than 2.

        """
        if num_waypoints < 2:
            raise ValueError("Number of waypoints must be at least 2.")

        # Generate waypoints using linear interpolation
        t = np.linspace(0, 1, num_waypoints)
        x = interp1d(t, [start[0], end[0]])
        y = interp1d(t, [start[1], end[1]])
        waypoints = np.column_stack([x(t), y(t)])

        return waypoints

    def find_optimal_path(
        self, start: np.ndarray, end: np.ndarray, obstacles: List[np.ndarray]
    ) -> np.ndarray:
        """
        Find the optimal path between two points while avoiding dynamic obstacles.

        This method uses a modified A* search algorithm to find the optimal path.

        Parameters
        ----------
        start : np.ndarray
            Starting position (x, y) of the path.

        end : np.ndarray
            Ending position (x, y) of the path.

        obstacles : List[np.ndarray]
            List of dynamic obstacle positions (x, y) to avoid.

        Returns
        -------
        np.ndarray
            Array of waypoints representing the optimal path.

        """
        # Update obstacle data and build KD-tree for efficient lookup
        self.obstacles = obstacles
        self.tree = KDTree(self.obstacles)

        # Initialize open and closed sets
        open_set = set()
        closed_set = set()

        # Initialize starting node
        start_node = Node(start, None, 0)
        open_set.add(start_node)

        # While the open set is not empty
        while open_set:
            # Get the current node with the lowest cost
            current_node = self.get_lowest_cost_node(open_set)

            # Check if we have reached the goal
            if np.allclose(current_node.position, end, atol=self.config["position_tolerance"]):
                logger.info("Optimal path found to the goal.")
                return self.reconstruct_path(current_node)

            # Generate child nodes and add them to the open set
            child_nodes = self.generate_child_nodes(current_node)
            for child_node in child_nodes:
                if child_node.position not in self.is_collision(child_node.position):
                    open_set.add(child_node)

            # Add current node to the closed set
            closed_set.add(current_node)
            open_set.remove(current_node)

        logger.warning("Could not find a valid path to the goal.")
        return None

    def get_lowest_cost_node(self, nodes: set) -> "Node":
        """
        Find the node with the lowest cost in the set of nodes.

        Parameters
        ----------
        nodes : set
            Set of nodes to search.

        Returns
        -------
        Node
            Node with the lowest cost.

        """
        lowest_cost_node = None
        for node in nodes:
            if (
                lowest_cost_node is None
                or node.total_cost < lowest_cost_node.total_cost
                or (
                    node.total_cost == lowest_cost_node.total_cost
                    and node.heuristic_cost < lowest_cost_node.heuristic_cost
                )
            ):
                lowest_cost_node = node

        return lowest_cost_node

    def generate_child_nodes(self, parent_node: "Node") -> List["Node"]:
        """
        Generate child nodes for the A* search algorithm.

        Parameters
        ----------
        parent_node : Node
            Parent node for which child nodes are generated.

        Returns
        -------
        List[Node]
            List of child nodes.

        """
        child_nodes = []
        for motion in self.config["valid_motions"]:
            new_position = parent_node.position + motion
            new_node = Node(
                new_position, parent_node, self.calculate_cost(new_position)
            )
            child_nodes.append(new_node)

        return child_nodes

    def calculate_cost(self, position: np.ndarray) -> float:
        """
        Calculate the cost function for a given position.

        The cost function includes the distance-based cost and the heuristic cost.

        Parameters
        ----------
        position : np.ndarray
            Position (x, y) for which the cost is calculated.

        Returns
        -------
        float
            Total cost at the given position.

        """
        distance_cost = np.linalg.norm(position - self.vehicle_model.goal_position)
        heuristic_cost = self.heuristic_function(position)
        total_cost = distance_cost + heuristic_cost

        return total_cost

    def heuristic_function(self, position: np.ndarray) -> float:
        """
        Heuristic function to estimate the cost from the current position to the goal.

        This function uses the Euclidean distance as a heuristic.

        Parameters
        ----------
        position : np.ndarray
            Position (x, y) for which the heuristic is calculated.

        Returns
        -------
        float
            Heuristic cost estimate.

        """
        return np.linalg.norm(position - self.vehicle_model.goal_position)

    def is_collision(self, position: np.ndarray) -> bool:
        """
        Check if a given position is in collision with any obstacles.

        Parameters
        ----------
        position : np.ndarray
            Position (x, y) to check for collision.

        Returns
        -------
        bool
            True if the position is in collision, False otherwise.

        """
        # Check for collision with static and dynamic obstacles
        collision = False
        for obstacle in self.config["static_obstacles"]:
            if np.linalg.norm(position - obstacle) < self.vehicle_model.radius:
                collision = True
                break

        if not collision:
            dist, idx = self.tree.query(position, k=1)
            if dist < self.vehicle_model.radius:
                collision = True

        return collision

    def reconstruct_path(self, end_node: "Node") -> np.ndarray:
        """
        Reconstruct the optimal path from the starting node to the end node.

        Parameters
        ----------
        end_node : Node
            Ending node of the optimal path.

        Returns
        -------
        np.ndarray
            Array of waypoints representing the optimal path.

        """
        path = [end_node.position]
        current_node = end_node.parent
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent

        path = np.array(path[::-1])

        return path

class Node:
    """
    Node class for the A* search algorithm.

    Attributes
    ----------
    position : np.ndarray
        Position (x, y) of the node.

    parent : Node
        Parent node.

    total_cost : float
        Total cost of the node (distance cost + heuristic cost).

    heuristic_cost : float
        Heuristic cost estimate from the node to the goal.

    """

    def __init__(self, position: np.ndarray, parent: "Node", total_cost: float):
        """
        Initialize the Node.

        Parameters
        ----------
        position : np.ndarray
            Position (x, y) of the node.

        parent : Node
            Parent node.

        total_cost : float
            Total cost of the node.

        """
        self.position = position
        self.parent = parent
        self.total_cost = total_cost
        self.heuristic_cost = self.heuristic_function(position)

    def heuristic_function(self, position: np.ndarray) -> float:
        """
        Heuristic function to estimate the cost from the current position to the goal.

        This function calculates the Euclidean distance to the goal.

        Parameters
        ----------
        position : np.ndarray
            Position (x, y) for which the heuristic is calculated.

        Returns
        -------
        float
            Heuristic cost estimate.

        """
        return np.linalg.norm(position - self.parent.vehicle_model.goal_position)

class VehicleModel:
    """
    Vehicle model class defining the dynamics and constraints of the automated vehicle.

    Attributes
    ----------
    length : float
        Length of the vehicle.

    width : float
        Width of the vehicle.

    height : float
        Height of the vehicle.

    max_acceleration : float
        Maximum acceleration of the vehicle.

    max_velocity : float
        Maximum velocity of the vehicle.

    goal_position : np.ndarray
        Goal position (x, y) that the vehicle is trying to reach.

    radius : float
        Radius of the vehicle for collision checking.

    """

    def __init__(
        self,
        length: float,
        width: float,
        height: float,
        max_acceleration: float,
        max_velocity: float,
        goal_position: np.ndarray,
    ):
        """
        Initialize the VehicleModel.

        Parameters
        ----------
        length : float
            Length of the vehicle.

        width : float
            Width of the vehicle.

        height : float
            Height of the vehicle.

        max_acceleration : float
            Maximum acceleration of the vehicle.

        max_velocity : float
            Maximum velocity of the vehicle.

        goal_position : np.ndarray
            Goal position (x, y) that the vehicle is trying to reach.

        """
        self.length = length
        self.width = width
        self.height = height
        self.max_acceleration = max_acceleration
        self.max_velocity = max_velocity
        self.goal_position = goal_position
        self.radius = np.sqrt(self.length**2 + self.width**2) / 2

# Example usage
if __name__ == "__main__":
    # Configure motion planner
    config = {
        "valid_motions": [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])],
        "static_obstacles": [np.array([10, 10]), np.array([5, 5])],
        "position_tolerance": 1e-3,
    }

    # Define vehicle model
    vehicle_model = VehicleModel(
        length=4,
        width=2,
        height=2,
        max_acceleration=1.0,
        max_velocity=10.0,
        goal_position=np.array([0, 0]),
    )

    # Initialize motion planner
    motion_planner = GlobalMotionPlanner(config, vehicle_model)

    # Define start and end positions
    start = np.array([0, 0])
    end = np.array([10, 10])

    # Generate reference trajectory
    reference_trajectory = motion_planner.generate_reference_trajectory(start, end, 10)
    print("Reference Trajectory:", reference_trajectory)

    # Define dynamic obstacles
    dynamic_obstacles = [np.array([5, 5]), np.array([7, 7])]

    # Find optimal path
    optimal_path = motion_planner.find_optimal_path(start, end, dynamic_obstacles)
    print("Optimal Path:", optimal_path)