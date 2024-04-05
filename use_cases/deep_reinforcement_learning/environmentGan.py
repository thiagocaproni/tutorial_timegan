import os
import numpy as np
import subprocess
import gymnasium as gym
from gymnasium import spaces

# Defining constants to help indexing data in DataFrames
FPS_INDEX = 0
BUFFER_INDEX = 1

TIMESTAMP_INDEX = 0
FIRST_SECOND = 0
FOURTH_SECOND = -1

BUFFER_SIZE_64 = 0
BUFFER_SIZE_32 = 1

class ControlPlaneEnvironment(gym.Env):
    """
    Custom environment for control plane actions.

    Attributes:
        action_space (gym.Space): The action space (discrete with 2 possible actions).
        reward (float): The current reward.
        done (bool): Flag indicating if the episode is done.
        current_state (object): The current state.
        next_state (object): The next state.
        buffer_size (list): List to store buffer sizes.
        actions_history (list): List to store action history.
        reward_history (list): List to store reward history.
        fps_history (list): List to store FPS history.
        lbo_history (list): List to store FPS history.
        received_dashStates (list): List to store received DASH states.
        received_intStates (list): List to store received INT states.

    Methods:
        take_action(action, intState, dashState):
            Takes an action based on the provided parameters and updates relevant state variables.
            Args:
                action (int): The action to be taken.
                intState (list): A DataFrame containing INT states.
                dashState (list): A DataFrame containing DASH states.
            Returns:
                tuple: A tuple containing the current state, next state, reward, done flag, and an empty dictionary (optional info).

        calculate_reward(buffer_action_step, buffer_reward_step, fps_reward_step):
            Calculates the reward score based on QoS level after the action taken.
            Args:
                buffer_action_step (float): The buffer action step.
                buffer_reward_step (float): The buffer reward step.
                fps_reward_step (float): The FPS reward step.
            Returns:
                None: The reward is stored in the instance variable 'self.reward'.
    """
    def __init__(self):
        """
        Initializes the environment.
        """
        self.action_space = spaces.Discrete(2)

        self.reward = 0
        self.done = False
        self.current_state = None
        self.next_state = None

        self.buffer_size = []
        self.actions_history = []
        self.reward_history = [0]
        self.fps_history = []
        self.lbo_history = []
        self.received_dashStates = []
        self.received_intStates = []


    def take_action(self, action, intState, dashState):
        """
        Takes an action based on the provided parameters and updates relevant state variables.

        Args:
            action (int): The action to be taken.
            intState (list): A DataFrame containing INT states.
            dashState (list): A DataFrame containing DASH states.

        Returns:
            tuple: A tuple containing the current state, next state, reward, done flag, and an empty dictionary (optional info).
        """

        # Verifying if the received action belongs the environment's action space
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Update the current state with the received INT state
        self.current_state = intState
        
        # Append the received INT state and DASH state to their respective lists
        self.received_intStates.append(intState)
        self.received_dashStates.append(dashState)
        
        # Append the received action to its respective list to keep track of the action history
        self.actions_history.append(action)
        
        # Update buffer size based on the action taken (this will be used to retreive the synthetic
        #  data from the correct files)
        if action == BUFFER_SIZE_64:
            self.buffer_size.append(64)
        elif action == BUFFER_SIZE_32:
            self.buffer_size.append(32)
        
        # Append FPS from the received DASH state to its respective list to keep track of the FPS history
        self.fps_history.append(dashState[FOURTH_SECOND][FPS_INDEX])

        # Append LBO from the received DASH state to its respective list to keep track of the FPS history
        self.lbo_history.append(dashState[FOURTH_SECOND][BUFFER_INDEX])
        
        # If two INT states have been received, it means that already have the current and next states
        # stored, hence we can now calculate the reward for the action taken
        if len(self.received_intStates) == 2:
            # state observed when the action was taken
            self.current_state = self.received_intStates[0]
            # resulting state after the taken action
            self.next_state = self.received_intStates[-1]    
            
            # dash state when the action was taken
            current_dash = self.received_dashStates[0]
            # dash state after the taken action
            next_dash = self.received_dashStates[-1]

            # You can uncomment the following lines and perform a sanity check to confirm whether the
            # current and next states are in the correct time window (4 seconds)

            #print("\ncurrent state timestamp (4th second): ", self.received_dashStates[0][FOURTH_SECOND][TIMESTAMP_INDEX])
            #print("next state timestamp (4th second): ", self.received_dashStates[-1][FOURTH_SECOND][TIMESTAMP_INDEX])
            #print("\n")

            # Calculate the reward based on LBO and FPS from the DASH state
            self.calculate_reward(current_dash[FOURTH_SECOND][BUFFER_INDEX], next_dash[FOURTH_SECOND][BUFFER_INDEX], next_dash[FOURTH_SECOND][FPS_INDEX])

            # Clear the received states for the next iteration    
            self.received_intStates = []
            self.received_dashStates = []
        
        # Return the experience that the agent gained by interacting with the environment
        return self.current_state, self.next_state, self.reward, self.done, {}


    def calculate_reward(self, buffer_action_step, buffer_reward_step, fps_reward_step):
        """
        Calculates the reward score based on QoS level after the action taken.

        Args:
            buffer_action_step (float): The buffer action step.
            buffer_reward_step (float): The buffer reward step.
            fps_reward_step (float): The FPS reward step.

        Returns:
            None: The reward is stored in the instance variable 'self.reward'.
        """

        # Checking whether the LBO improved after the action taken
        if buffer_reward_step > buffer_action_step:
            # If the next state's LBO is greater than 30, assign a reward of 2
            if buffer_reward_step > 30:
                self.reward = 2
            # Otherwise, check the next state's FPS and assign a reward score accordingly
            elif buffer_reward_step < 30:
                if fps_reward_step == 30:
                    self.reward = 1
                elif fps_reward_step == 24:
                    self.reward = .5
                else:
                    self.reward = .1
        
        # Checking whether the LBO retarded after the action taken
        if buffer_reward_step < buffer_action_step:
            # If the next state's LBO is greater than 30, assign a reward of 2
            if buffer_reward_step > 30:
                self.reward = 2
            # Otherwise, check the next state's FPS and assign a reward score accordingly
            elif buffer_reward_step < 30:
                if fps_reward_step == 30:
                    self.reward = 1
                elif fps_reward_step == 24:
                    self.reward = .5
                else:
                    self.reward = -2
        
        # Append the calculated reward to the reward history
        self.reward_history.append(self.reward)
    