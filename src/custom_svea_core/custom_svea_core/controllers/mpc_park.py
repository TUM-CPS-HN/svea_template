#!/usr/bin/env python3

import casadi as ca
import numpy as np
from rclpy.node import Node

class MPCParkingController:

    def __init__(self, node: Node) -> None:
        # Core Parameters
        self._node = node

        # Parameters are defined locally for this controller instance.
        self.N = 30 # horizon

        # The time step in which the optimization problem is divided (unit [s]).
        # TODO: Can we put a time limit on the solve?
        self.dt = ca.DM(0.4)

        # we convert to dense matrix to allow symbolic operations.
        # We should increase the weight on the orientation error to ensure the vehicle aligns with the wall.
        self.Q1 = ca.DM([
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 17.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ])

        self.w4 = 5
        self.w5 = 2
        self.w6 = 5
        self.w1 = self.N * self.w4 # so that the cost of the final state is always more important than ALL intermediate ones.
        self.w2 = self.N * self.w5
        self.w3 = self.N * self.w6
        self.w7 = 3
        self.w8 = 3

        # Model Parameters
        self.min_steering = np.radians(-40.0)
        self.max_steering = np.radians(40.0)

        self.min_steering_rate = np.radians(-35.0)
        self.max_steering_rate = np.radians(35.0)

        self.min_velocity = -0.4
        self.max_velocity = 0.0

        self.min_acceleration = -0.2
        self.max_acceleration = 0.2

        # Wheelbase of the vehicle (unit [m]).
        self.L = ca.DM(0.34)

        # Front and rear safety circles offsets.
        self.f_circle_offset = ca.DM(0.34)
        self.r_circle_offset = ca.DM(0.15)
        self.circle_radius = ca.DM(0.09)

        # # Wall alignment tolerance for the final orientation (unit [rad]).
        # self.eps_wall = ca.DM(0.1)
        
        # Setup CasADi
        self.opti = ca.Opti()

        self.define_state_and_control_variables()
        self.set_objective_function()
        self.set_state_constraints()
        self.set_control_input_constraints()
        self.set_solver_options()
        self.sol = None # Store the solution after optimization

    def compute_control(self, state, target_on_wall, wall_line):
        """
        Compute the control actions using MPC.
        
        :param state: Current state of the vehicle [x, y, theta, v, delta]
        :param target_spot: Target spot information [x, y]
        :param wall_line: Wall line information [A, B, C]
        """
        state = self.bound_initial_state(state)

        # Transform base_link to rear_axle
        yaw = state[2]
        state[0] = state[0] - 0.15 * np.cos(yaw)
        state[1] = state[1] - 0.15 * np.sin(yaw)

        A, B, C = wall_line
        
        # Car's heading vector
        car_vector = np.array([np.cos(yaw), np.sin(yaw)])
        
        # Wall's direction vector (one of the two possibilities)
        wall_vector = np.array([-B, A])
        
        # If the dot product is negative, the angle between them is > 90 degrees.
        if np.dot(car_vector, wall_vector) < 0:
            wall_vector = -wall_vector

        raw_wall_angle = np.arctan2(wall_vector[1], wall_vector[0])
        angle_diff = raw_wall_angle - yaw
        shortest_turn = np.arctan2(np.sin(angle_diff), np.cos(angle_diff)) 
        
        wall_angle = yaw + shortest_turn # This is to avoid the problems assiciated with angle wrapping.

        full_target = [
            target_on_wall[0], 
            target_on_wall[1], 
            wall_angle, 
            0.0,
            0.0
        ]

        self.opti.set_value(self.x_init, state)
        self.opti.set_value(self.x_target, full_target)
        self.opti.set_value(self.wall_line, wall_line)
        self.opti.set_value(self.wall_unit_vector, wall_vector / np.linalg.norm(wall_vector))

        try:
            self.sol = self.opti.solve()
            
            # Extract steering for the NEXT state, since the actuation interface needs this.
            steering_cmd = self.sol.value(self.x[4, 1])
            
            # Extract velocity for the NEXT state, let see if this makes it more stable. It did.
            velocity_cmd = self.sol.value(self.x[3, 1])
            
            return steering_cmd, velocity_cmd
            
        except RuntimeError as e:
            self._node.get_logger().error(f"IPOPT Failed: {e}")
            return 0.0, 0.0


    def get_optimal_control(self, all=True):
        """
        This method returns the optimal control computed by the mpc.
        :param all: Flag to indicate if all of the optimal control for the 
            whole prediction should	be returned or only the first step.
        :type all: bool
        :return: The optimal control computed by the mpc controller.
        :rtype: NumPy array
        """
        if self.sol is not None:
            u_opt = np.array(self.sol.value(self.u))
            if all:
                return u_opt
            else:
                return u_opt[:,0]
        else:
            return None
        
    def get_optimal_states(self):
        """
        This method returns the optimal states computed by the mpc controller
        for the whole prediction horizon.
        :return: The optimal states for the whole prediction horizon.
        :rtype: NumPy array
        """
        if self.sol is not None:
            return np.array(self.sol.value(self.x))   # Note: to get only optimized one: self.sol.value(self.x[:, :self.N])
        else: 
            return None

    def define_state_and_control_variables(self):
        # Define state and control variables
        self.x = self.opti.variable(5, self.N + 1)  # state = [x, y, theta, v, steering]
        self.u = self.opti.variable(2, self.N)      # input = [acceleration, steering_rate]

        self.x_init = self.opti.parameter(5)             # Initial state
        self.x_target = self.opti.parameter(5)           # Target parking state, has the wall angle too.
        self.wall_line = self.opti.parameter(3)        # Linear equation of the wall: ax + by + c = 0, wall_line = [a, b, c]
        self.wall_unit_vector = self.opti.parameter(2) # unit vector in the direction of the wall.

    def set_objective_function(self):
        # Define the objective function: The first approach: 
        # J = (x[N]-x_target)^T Q1 (x[N]-x_target)
        # If you want to add more terms to the objective function, just do: self.objective += <your_term_here>
        # self.objective = ca.mtimes([(self.x[:, self.N] - self.x_target).T, self.Q1, (self.x[:, self.N] - self.x_target)]) 

        # The second approach
        # cost of the longitudinal distance
        self.objective = self.w1 * ca.mtimes([(self.x[0:2, self.N] - self.x_target[0:2]).T, self.wall_unit_vector @ self.wall_unit_vector.T, (self.x[0:2, self.N] - self.x_target[0:2])])

        # cost of the lateral distance
        self.objective += self.w2 * (self.wall_line[0] * self.x[0, self.N] + self.wall_line[1] * self.x[1, self.N] + self.wall_line[2])**2 / (self.wall_line[0]**2 + self.wall_line[1]**2 + 1e-6)

        # cost of orientation error
        self.objective += self.w3 * (self.x[2, self.N] - self.x_target[2])**2

        # cost of intermediate errors from the target, to make the car not take forever.
        for k in range(1, self.N):
            self.objective += self.w4 * (ca.mtimes([(self.x[0:2, k] - self.x_target[0:2]).T, self.wall_unit_vector @ self.wall_unit_vector.T, (self.x[0:2, k] - self.x_target[0:2])]))
            # Removing these was right, they made the car go in wrong directions, and the first one is sufficient to make the car go faster
            # self.objective += self.w5 * (self.wall_line[0] * self.x[0, k] + self.wall_line[1] * self.x[1, k] + self.wall_line[2])**2 / (self.wall_line[0]**2 + self.wall_line[1]**2 + 1e-6) # think about removing this.
            # self.objective += self.w6 * (self.x[2, k] - self.x_target[2])**2 
            # self.objective += self.w7 * self.u[0, k]**2 + self.w8 * self.u[1, k]**2 # to stop the car from jittering when it reaches the goal. I think I will simply replace this by a stopping condition at the main script.

        # Specify type of optimization problem
        self.opti.minimize(self.objective)

    def set_state_constraints(self):
        # Initial state constraint
        self.opti.subject_to(self.x[:, 0] == self.x_init)

        # calculate once
        A = self.wall_line[0]
        B = self.wall_line[1]
        C = self.wall_line[2]
        denominator = A**2 + B**2 + 1e-6 # avoid division by zero

        # Vehicle dynamics constraints - Simple kinematic bycicle model
        for k in range(self.N):
            x_next = self.x[0, k] + self.dt * self.x[3, k] * ca.cos(self.x[2, k])                   # x_k+1 = x_k + dt * v_k * cos(theta_k)
            y_next = self.x[1, k] + self.dt * self.x[3, k] * ca.sin(self.x[2, k])                   # y_k+1 = y_k + dt * v_k * sin(theta_k)
            theta_next = self.x[2, k] + self.dt * (self.x[3, k] / self.L) * ca.tan(self.x[4, k])    # theta_k+1 = theta_k + dt * v_k * tan(delta_k) / L
            v_next = self.x[3, k] + self.dt * self.u[0, k]                                          # v_k+1 = v_k + dt * a_k
            delta_next = self.x[4, k] + self.dt * self.u[1, k]                                      # delta_k+1 = delta_k + dt * steering_rate_k

            self.opti.subject_to(self.x[0, k + 1] == x_next)
            self.opti.subject_to(self.x[1, k + 1] == y_next)
            self.opti.subject_to(self.x[2, k + 1] == theta_next)
            self.opti.subject_to(self.x[3, k + 1] == v_next)
            self.opti.subject_to(self.x[4, k + 1] == delta_next)

            # Velocity constraints
            self.opti.subject_to(self.x[3, k+1] <= self.max_velocity)
            self.opti.subject_to(self.x[3, k+1] >= self.min_velocity)

            # Steering angle constraints
            self.opti.subject_to(self.x[4, k+1] <= self.max_steering)
            self.opti.subject_to(self.x[4, k+1] >= self.min_steering)

            # Wall direction constraints, also avoiding using abs()
            self.opti.subject_to(self.x[2, k+1] - self.x_target[2] <= np.pi/2)
            self.opti.subject_to(self.x[2, k+1] - self.x_target[2] >= -np.pi/2)

            # Rear circle (centered directly on the rear axle)
            dist_rear = ((A * self.x[0, k+1] + B * self.x[1, k+1] + C)**2) / denominator
            self.opti.subject_to(dist_rear >= self.circle_radius**2)
            
            # Front circle (shifted forward by wheelbase L)
            dist_two_centers = self.r_circle_offset + self.f_circle_offset # distance between the two circle centers
            front_x = self.x[0, k+1] + dist_two_centers * ca.cos(self.x[2, k+1])
            front_y = self.x[1, k+1] + dist_two_centers * ca.sin(self.x[2, k+1])
            dist_front = ((A * front_x + B * front_y + C)**2) / denominator
            self.opti.subject_to(dist_front >= self.circle_radius**2)

        self.opti.subject_to(self.x[3, self.N] == 0.0)

        # # wall alignment condition. Ensure final orientation aligns with wall angle
        # self.opti.subject_to(self.x[2, self.N] - self.x_target[2] <= self.eps_wall)
        # self.opti.subject_to(self.x[2, self.N] - self.x_target[2] >= -self.eps_wall)

    def set_control_input_constraints(self):
        # Input constraints (acceleration, steering rate)
        for k in range(self.N):
            self.opti.subject_to(self.min_acceleration <= self.u[0, k])
            self.opti.subject_to(self.u[0, k] <= self.max_acceleration)

            # Steering rate constraint
            self.opti.subject_to(self.min_steering_rate <= self.u[1, k])
            self.opti.subject_to(self.u[1, k] <= self.max_steering_rate)

    def set_solver_options(self):
        # Set solver options
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

    # Keep an eye on this function. This might be useful against noise and weird initial positions.
    def bound_initial_state(self,state):
        """
        This method checks if the initial state provided to the mpc, which could come from the localization stack,
        is within the allowed bounds. If not, it clamps it to make the optimization problem feasible.
        """
        # Ensure the velocity is within the specified bounds
        clamped_velocity = max(self.min_velocity, min(state[3], self.max_velocity))
        
        # Create a new state with the clamped velocity
        bounded_state = state.copy()
        bounded_state[3] = clamped_velocity
        
        return bounded_state
    
    # def set_new_prediction_horizon(self, new_horizon):
    #     """
    #     Dynamically adjust the active horizon for the optimization problem.
    #     When the horizon is reduced, you simply freeze unused variables and update the reference trajectory 
    #     and state for the active part of the horizon. The solver will then only optimize over the active steps.
    #     """
    #     self.N = new_horizon
        
    #     # Redefine objective function with new horizon.
    #     self.set_objective_function()

    # I think no need for this in the parking task.
    # def compute_state_error(self, x, x_ref):
    #     """
    #     Computes the state error between the current state and the reference state.
    #     Adjusts the yaw error to account for angle wrapping in the range [-π, π].
    #     """
    #     state_error = x[0:4] - x_ref[0:4]
    #     yaw_diff = x[2] - x_ref[2]
    #     state_error[2] = ca.atan2(ca.sin(yaw_diff), ca.cos(yaw_diff))  # Minimum yaw error 
    #     return state_error

    # def set_new_weight_matrix(self, matrix_name, new_value):
    #     """
    #     Dynamically adjust one of the weight matrices.
    #     Check if the matrix exists as an attribute and if so, update it.
    #     """
    #     if hasattr(self, matrix_name):
    #         try:
    #             # Overwrite with the new dense matrix
    #             setattr(self, matrix_name, ca.DM(new_value))
    #             # Redefine objective function with updated matrix.
    #             self.set_objective_function()
    #         except Exception as e:
    #             print(f"Failed to update {matrix_name}: {e}")
    #     else:
    #         print(f"Matrix {matrix_name} does not exist in MPC class.")

    # def reset_parameters(self):
    #     """
    #     Reset the core parameters and weight matrices of the MPC instance to their initial values.
    #     Useful for restoring values to their original state after runtime modifications.
    #     """
    #     self.N = self.N  # Reset to max horizon

    #     # Reset weight matrices
    #     self.Q1 = ca.DM(np.array(self.Q1_list).reshape((4, 4)))
    #     self.Q2 = ca.DM(np.array(self.Q2_list).reshape((2, 2)))
    #     self.Q3 = ca.DM(np.array(self.Q3_list).reshape((2, 2)))
    #     self.Qf = ca.DM(np.array(self.Qf_list).reshape((4, 4)))
    #     self.Qv = ca.DM(self.Qv_num)
    #     # Reset objective function with initial values.
    #     self.set_objective_function()