import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.spatial import distance
from simulation_environment.config import SimulationConfig
from simulation_environment.exceptions import SimulationError
from simulation_environment.models import Vehicle, Road, Obstacle
from simulation_environment.utils import calculate_velocity, calculate_distance

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationEnvironment:
    """
    Simulation environment for automated vehicles.

    Attributes:
        config (SimulationConfig): Simulation configuration.
        vehicles (List[Vehicle]): List of vehicles in the simulation.
        roads (List[Road]): List of roads in the simulation.
        obstacles (List[Obstacle]): List of obstacles in the simulation.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation environment.

        Args:
            config (SimulationConfig): Simulation configuration.
        """
        self.config = config
        self.vehicles = []
        self.roads = []
        self.obstacles = []

    def initialize_simulation(self):
        """
        Initialize the simulation environment.

        Raises:
            SimulationError: If the simulation configuration is invalid.
        """
        try:
            # Validate simulation configuration
            self.config.validate()

            # Create roads
            for road in self.config.roads:
                self.roads.append(Road(road))

            # Create vehicles
            for vehicle in self.config.vehicles:
                self.vehicles.append(Vehicle(vehicle))

            # Create obstacles
            for obstacle in self.config.obstacles:
                self.obstacles.append(Obstacle(obstacle))

            logger.info("Simulation environment initialized.")
        except SimulationError as e:
            logger.error(f"Simulation initialization failed: {e}")
            raise

    def run_simulation(self):
        """
        Run the simulation.

        Raises:
            SimulationError: If the simulation fails.
        """
        try:
            # Initialize simulation
            self.initialize_simulation()

            # Run simulation loop
            while self.config.simulation_time > 0:
                # Update vehicle positions
                for vehicle in self.vehicles:
                    vehicle.update_position()

                # Check for collisions
                for vehicle in self.vehicles:
                    for obstacle in self.obstacles:
                        if distance.euclidean(vehicle.position, obstacle.position) < vehicle.radius + obstacle.radius:
                            logger.error("Collision detected!")
                            raise SimulationError("Collision detected!")

                # Update simulation time
                self.config.simulation_time -= 1

                # Log simulation progress
                logger.info(f"Simulation time: {self.config.simulation_time} seconds")

            logger.info("Simulation completed.")
        except SimulationError as e:
            logger.error(f"Simulation failed: {e}")
            raise

class Road:
    """
    Road model.

    Attributes:
        id (int): Road ID.
        length (float): Road length.
        width (float): Road width.
    """

    def __init__(self, config: Dict):
        """
        Initialize the road model.

        Args:
            config (Dict): Road configuration.
        """
        self.id = config["id"]
        self.length = config["length"]
        self.width = config["width"]

class Vehicle:
    """
    Vehicle model.

    Attributes:
        id (int): Vehicle ID.
        position (Tuple[float, float]): Vehicle position.
        velocity (float): Vehicle velocity.
        radius (float): Vehicle radius.
    """

    def __init__(self, config: Dict):
        """
        Initialize the vehicle model.

        Args:
            config (Dict): Vehicle configuration.
        """
        self.id = config["id"]
        self.position = (config["position"][0], config["position"][1])
        self.velocity = config["velocity"]
        self.radius = config["radius"]

    def update_position(self):
        """
        Update the vehicle position based on its velocity.
        """
        self.position = (self.position[0] + self.velocity, self.position[1])

class Obstacle:
    """
    Obstacle model.

    Attributes:
        id (int): Obstacle ID.
        position (Tuple[float, float]): Obstacle position.
        radius (float): Obstacle radius.
    """

    def __init__(self, config: Dict):
        """
        Initialize the obstacle model.

        Args:
            config (Dict): Obstacle configuration.
        """
        self.id = config["id"]
        self.position = (config["position"][0], config["position"][1])
        self.radius = config["radius"]

class SimulationConfig:
    """
    Simulation configuration.

    Attributes:
        simulation_time (int): Simulation time.
        roads (List[Dict]): Road configurations.
        vehicles (List[Dict]): Vehicle configurations.
        obstacles (List[Dict]): Obstacle configurations.
    """

    def __init__(self, config: Dict):
        """
        Initialize the simulation configuration.

        Args:
            config (Dict): Simulation configuration.
        """
        self.simulation_time = config["simulation_time"]
        self.roads = config["roads"]
        self.vehicles = config["vehicles"]
        self.obstacles = config["obstacles"]

    def validate(self):
        """
        Validate the simulation configuration.

        Raises:
            SimulationError: If the simulation configuration is invalid.
        """
        if self.simulation_time <= 0:
            raise SimulationError("Simulation time must be greater than 0.")

        for road in self.roads:
            if road["length"] <= 0 or road["width"] <= 0:
                raise SimulationError("Road length and width must be greater than 0.")

        for vehicle in self.vehicles:
            if vehicle["velocity"] <= 0 or vehicle["radius"] <= 0:
                raise SimulationError("Vehicle velocity and radius must be greater than 0.")

        for obstacle in self.obstacles:
            if obstacle["radius"] <= 0:
                raise SimulationError("Obstacle radius must be greater than 0.")

class SimulationError(Exception):
    """
    Simulation error.
    """
    pass

if __name__ == "__main__":
    # Create simulation configuration
    config = SimulationConfig({
        "simulation_time": 100,
        "roads": [
            {"id": 1, "length": 100, "width": 10},
            {"id": 2, "length": 50, "width": 5}
        ],
        "vehicles": [
            {"id": 1, "position": [0, 0], "velocity": 5, "radius": 2},
            {"id": 2, "position": [50, 0], "velocity": 3, "radius": 1}
        ],
        "obstacles": [
            {"id": 1, "position": [20, 0], "radius": 1}
        ]
    })

    # Create simulation environment
    simulation = SimulationEnvironment(config)

    # Run simulation
    simulation.run_simulation()