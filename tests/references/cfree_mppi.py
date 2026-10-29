"""Unmodified upstream methods used only for optional Torch parity tests.

Copyright (c) 2025 Yukang Cao. MIT; see reversible/C_FREE_UNIFORM_LICENSE.txt.
Source file names and SHA256 are recorded in docs/reversible-mppi.md.
"""
import torch

class ReferenceCost:
    def _calculate_trajectory_costs(self, robot_frame_trajectories: torch.Tensor, goal_tensor: torch.Tensor) -> torch.Tensor:
            """
            (Fully Vectorized) Calculates the total cost for a batch of trajectories in the robot frame.
            Implements the logic of early exit (cost accumulation stops after goal reach or collision).
            robot_frame_trajectories shape: (T+1, K, 3)
            goal_tensor shape: (2,) for position or (3,) for position + orientation
            Returns shape: (K,)
            """
            # Input shape: (T+1, K, 3)
            positions = robot_frame_trajectories[..., :2] # (T+1, K, 2)
            orientations = robot_frame_trajectories[..., 2] # (T+1, K)

            # 1. Calculate costs at every timestep (Stateless calculations)
            # 1a. Distance costs (T+1, K)
            # Calculate squared distance to goal for all trajectories at all timesteps.
            dist_to_goal_squared = torch.sum((positions - goal_tensor[:2])**2, dim=2)
            distance_costs = dist_to_goal_squared * self.dist_weight

            # 1b. Orientation costs (T+1, K) - if enabled
            orientation_costs = torch.zeros_like(distance_costs)
            if hasattr(self.config, 'include_orientation_cost') and self.config['include_orientation_cost']:
                # Goal orientation (robot frame goal orientation is 0 when facing +X)
                goal_orientation = 0.0 if goal_tensor.shape[0] == 2 else goal_tensor[2].item()
            
                # Calculate angle difference with proper wrapping
                angle_diff = orientations - goal_orientation
                # Wrap angle difference to [-π, π]
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                # Square the angle difference and apply weight
                orientation_costs = (angle_diff ** 2) * self.config['orientation_cost_weight']

            # 1c. Obstacle costs (T+1, K)
            # Use the optimized, vectorized function which utilizes the pre-convolved costmap.
            obstacle_costs_raw = self._calculate_robot_frame_costmap_cost(positions)

            # 2. Determine stateful masks (Collision and Goal Reached)
            # 2a. Collision detection at each timestep (T+1, K)
            collision_threshold = self.robot_footprint_area * self.collision_occupancy_ratio
            # Boolean mask indicating if a collision occurred exactly at time t.
            collided_at_t = obstacle_costs_raw > collision_threshold

            # 2b. Goal reached detection at each timestep (T+1, K)
            goal_tolerance_squared = self.goal_tolerance ** 2
            # Position tolerance check
            position_reached = dist_to_goal_squared <= goal_tolerance_squared
        
            # Orientation tolerance check (if orientation cost is enabled)
            orientation_reached = torch.ones_like(position_reached)  # Default: always satisfied
            if hasattr(self.config, 'include_orientation_cost') and self.config['include_orientation_cost']:
                goal_orientation = 0.0 if goal_tensor.shape[0] == 2 else goal_tensor[2].item()
                angle_diff = orientations - goal_orientation
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                angle_tolerance = getattr(self.config, 'goal_angle_tolerance', self.goal_tolerance)
                orientation_reached = torch.abs(angle_diff) <= angle_tolerance
        
            # Boolean mask indicating if the goal was reached exactly at time t (both position and orientation)
            reached_at_t = position_reached & orientation_reached

            # 3. Propagate masks forward in time (Stateful logic vectorization)
            # If an event occurs at time t, the mask must remain true for all t' > t.
            # use cumulative maximum (cummax) along the time dimension (dim=0)
            # This acts as a cumulative OR operation. Using .byte() is efficient.

            # (T+1, K). True if collided at or before t
            # .values extracts the tensor from the (values, indices) tuple returned by cummax
            collision_mask = torch.cummax(collided_at_t.byte(), dim=0).values.bool()
            # (T+1, K). True if reached goal at or before t
            goal_reached_mask = torch.cummax(reached_at_t.byte(), dim=0).values.bool()

            # 4. Apply masks to costs
            # Combined termination mask (T+1, K)
            is_terminated = collision_mask | goal_reached_mask

            # 4a. Apply termination mask to distance costs
            # If terminated (collided or reached goal) at or before t, the distance cost at t is zeroed out
            distance_costs_masked = torch.where(
                is_terminated,
                torch.zeros_like(distance_costs),
                distance_costs
            )

            # 4a.5. Apply termination mask to orientation costs
            # If terminated (collided or reached goal) at or before t, the orientation cost at t is zeroed out
            orientation_costs_masked = torch.where(
                is_terminated,
                torch.zeros_like(orientation_costs),
                orientation_costs
            )

            # 4b. Process and apply masks to obstacle costs
            # Calculate the scaled obstacle cost (used when not in a hard collision)
            scaled_obstacle_costs = obstacle_costs_raw * (self.obs_penalty / self.robot_footprint_area)

            # Apply collision mask: If collided at or before t, use full penalty, otherwise use scaled cost
            obstacle_costs_penalized = torch.where(
                collision_mask,
                torch.full_like(scaled_obstacle_costs, self.obs_penalty),
                scaled_obstacle_costs
            )

            # Apply goal reached mask: If goal reached at or before t, zero out obstacle cost (early exit)
            obstacle_costs_masked = torch.where(
                goal_reached_mask,
                torch.zeros_like(obstacle_costs_penalized),
                obstacle_costs_penalized
            )

            # 5. Calculate Total Cost (Sum over time dimension)
            # (T+1, K) -> (K,)
            total_costs = torch.sum(distance_costs_masked + orientation_costs_masked + obstacle_costs_masked, dim=0)

            # 6. Terminal costs
            # Terminal costs apply only for trajectories that never reached the goal AND were always safe.

            # The final state of the masks (at T) tells us if the event ever happened.
            ever_collided = collision_mask[-1] # (K,)
            ever_reached_goal = goal_reached_mask[-1] # (K,)

            is_safe_and_unsuccessful = ~(ever_reached_goal | ever_collided)

            # Calculate terminal distance based on the very last position (already calculated)
            terminal_distances_squared = dist_to_goal_squared[-1]
            terminal_costs = is_safe_and_unsuccessful.float() * terminal_distances_squared * self.terminal_weight
            total_costs += terminal_costs
            return total_costs

class ReferenceUpdate:
    def _run_mppi_optimization(self, U_nominal, current_state, goal_state):
            """
            Runs the MPPI optimization loop in PyTorch.
            Implements the update rule matching reference_mppi.py (Weighted Effective Noise).
            """
            # Setup for robot frame calculations
            robot_frame_initial_state = torch.zeros(3, device=self.device, dtype=torch.float32)
            robot_frame_goal = self._transform_goal_to_robot_frame(goal_state, current_state[:3])
            goal_tensor = torch.from_numpy(robot_frame_goal).float().to(self.device)

            # Control limits for clamping (Tensors for broadcasting)
            # Shape (1, 1, 2)
            min_ctrl = torch.tensor([float(self.vrange[0]), float(self.wrange[0])], device=self.device).view(1, 1, 2)
            max_ctrl = torch.tensor([float(self.vrange[1]), float(self.wrange[1])], device=self.device).view(1, 1, 2)

            # MPPI Iteration Loop
            for iteration in range(self.mppi_iterations):
                # 1. Generate Noise (T, K, 2)
                noise = self._generate_noise_torch()

                # 2. Perturb nominal controls (T, K, 2)
                perturbed_controls = U_nominal.unsqueeze(1) + noise

                # Clamp controls to physical limits (Vectorized)
                perturbed_controls_clamped = torch.max(torch.min(perturbed_controls, max_ctrl), min_ctrl)

                # 3. Rollout trajectories (using clamped controls) (T+1, K, 3)
                mppi_trajectories = self._rollout_full_controls_torch(perturbed_controls_clamped, robot_frame_initial_state)

                # 4. Calculate costs (K,)
                costs = self._calculate_trajectory_costs(mppi_trajectories, goal_tensor)

                # 5. Weight trajectories (Importance Sampling)
                weights = torch.softmax(-(1.0 / self.lambda_weight) * costs, dim=0)

                # 6. Update Nominal Control (Standard MPPI Update)
                # Use the raw sampled noise for the update, ignoring clamping effects.
                noise_to_use = noise

                # Reshape weights: (K,) -> (1, K, 1)
                weights_reshaped = weights.unsqueeze(0).unsqueeze(2)

                # Calculate weighted perturbations
                # Sum((1, K, 1) * (T, K, 2)) along dim 1 -> (T, 2)
                perturbations = torch.sum(weights_reshaped * noise_to_use, dim=1)

                # Update nominal control
                U_nominal = U_nominal + perturbations

                # Final clamping (safety measure)
                U_nominal = torch.max(torch.min(U_nominal, max_ctrl.view(1, 2)), min_ctrl.view(1, 2))

            # Return the final optimized trajectory and the trajectories from the LAST iteration
            return U_nominal, mppi_trajectories
