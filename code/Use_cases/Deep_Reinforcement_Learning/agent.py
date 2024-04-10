import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from numpy.random import seed
seed(1)

import tensorflow as tf
tf.random.set_seed(1)

import random
random.seed(1)

from collections import deque
from random import sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2

from sklearn import preprocessing

class DDQN:
    def __init__(self, state_dim, num_actions, learning_rate, gamma,
                 epsilon_start, epsilon_end, epsilon_decay_steps,
                 epsilon_exponential_decay, replay_capacity,
                 architecture, l2_reg, tau, batch_size,
                 minimum_experience_memory, initialization,
                 online_network_filepath, target_network_filepath,
                 experiment_type):
        """
        Initializes an RL agent.

        Args:
            state_dim (int): Dimensionality of the state space.
            num_actions (int): Number of possible actions.
            learning_rate (float): Learning rate for optimization.
            gamma (float): Discount factor for future rewards.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Final exploration rate.
            epsilon_decay_steps (int): Number of steps for epsilon decay.
            epsilon_exponential_decay (float): Exponential decay factor for epsilon.
            replay_capacity (int): Maximum capacity of the experience replay buffer.
            architecture (tuple): Number of layers and units per layer of the neural network.
            l2_reg (float): L2 regularization strength.
            tau (int): Target network update rate.
            batch_size (int): Batch size for training.
            minimum_experience_memory (int): Minimum experience required before training.
            initialization (str): Initialization method (standard or pretrained).
            online_network_filepath (str): Filepath for loading online network weights.
            target_network_filepath (str): Filepath for loading target network weights.
            experiment_type (int): Type of RL experiment (1 - train the DQN, 2 - inference only,
                                   3 - transfer learning with weight freezing).

        Attributes:
            state_dim (int): Dimensionality of the state space.
            num_actions (int): Number of possible actions.
            experience (deque): Experience replay buffer.
            learning_rate (float): Learning rate for optimization.
            gamma (float): Discount factor for future rewards.
            architecture (tuple): Neural network architecture type.
            l2_reg (float): L2 regularization strength.
            minimum_experience_memory (int): Minimum experience required before training.
            initialization (str): Initialization method (standard or pretrained).
            online_network_filepath (str): Filepath for loading online network weights.
            target_network_filepath (str): Filepath for loading target network weights.
            epsilon (float): Initial probability to take random actions.
            epsilon_decay_steps (int): Number of steps for epsilon linear decay.
            epsilon_decay (float): Epsilon's linear decay rate per step.
            epsilon_exponential_decay (float): Exponential decay factor for epsilon.
            epsilon_history (list): History of epsilon values.
            total_steps (int): Total number of steps taken.
            train_steps (int): Number of training steps.
            episodes (int): Total number of episodes.
            episode_length (int): Length of the current episode.
            train_episodes (int): Number of episodes used for training.
            steps_per_episode (list): Steps taken per episode.
            episode_reward (float): Accumulated reward in the current episode.
            rewards_history (list): History of episode rewards.
            batch_size (int): Batch size for training.
            tau (float): Target network update rate.
            losses (list): Training losses.
            q_values (list): Q-values during training.
            idx (Tensor): Tensor for batch indices.
            train (bool): Flag indicating whether the agent is in training mode.
            inference_time (list): Time taken for inference per step.
            experiment_type (str): Type of RL experiment.
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg
        self.minimum_experience_memory = minimum_experience_memory

        self.initialization = initialization
        self.online_network_filepath = online_network_filepath
        self.target_network_filepath = target_network_filepath

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.q_values = []
        self.idx = tf.range(batch_size)
        self.train = True
        self.inference_time = []

        self.experiment_type = experiment_type

        # Verify the DQN initialization type
        if self.initialization == 'standard':
            # If the initialization method is 'standard', build new models
            # for the online and target networks, respectively, and clone
            # the online network weights to the target network 
            self.online_network = self.build_model()
            self.target_network = self.build_model(trainable=False)

            self.update_target()
        # If the initialization method is 'pretrained', load the pretrained
        # online and target networks from the filepaths provided
        elif self.initialization == 'pretrained':
            self.load_models()


    def build_model(self, trainable=True):
        """
        Builds a neural network model for Q-value estimation.

        Args:
            trainable (bool): Whether the model's weights are trainable (default: True).

        Returns:
            Sequential: A Keras Sequential model representing the Q-value estimator.
        """

        # Initialize an empty list to store layers
        layers = []
        # Get the number of layers from self.architecture
        n = len(self.architecture)

        # Iterate over each layer and create dense layers with the ReLu activation function
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None, # Add an input layer only if this is the 1st iteration
                                activation='relu',
                                #kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        
        # Add the output layer with the linear activation function
        layers.append(Dense(units=self.num_actions,
                            activation='linear',
                            trainable=trainable,
                            name='Output'))
        
        # Create a Sequential model using the defined layers
        model = Sequential(layers)

        # Configure the optimizer (Stochastic Gradient Descent)
        sgd = SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        
        # Compile the model with the mean squared error loss function
        model.compile(loss='mean_squared_error',
                      optimizer=sgd)

        # Print a summary of the model architecture
        model.summary()
        
        # Return a Keras Sequential model representing the Q-value estimator (online/target)
        return model
    

    def load_models(self):
        """
        Loads neural network models for online and target networks.

        Attributes:
            self.online_network: Neural network for online Q-value estimation.
            self.target_network: Neural network for target Q-value estimation.
            self.online_network_filepath (str): Filepath for loading online network weights.
            self.target_network_filepath (str): Filepath for loading target network weights.
            self.experiment_type (int): Type of RL experiment (1, 2, or 3).

        Returns:
            None: The online and target networks are stored in the self.online_network and
                  self.target_network attributes, respectively.
        """

        # If the experiment_type is 1, the Q-network will be loaded and updated throughout the new experiment

        # Load the online network
        self.online_network = tf.keras.saving.load_model(self.online_network_filepath)
        # Print a summary of the model architecture
        self.online_network.summary()

        # Load the target network
        self.target_network = tf.keras.saving.load_model(self.target_network_filepath)
        # Print a summary of the model architecture
        self.target_network.summary()

        # Synchronize the target network
        self.update_target()

        # If the experiment_type is 2, the Q-network will only make predictions throughout the new experiment, not learning
        if self.experiment_type == 2:
            # Freeze the Q-network layers (all layers)
            print("\n=================================================================\n")
            
            print("Freezing the q-network layers weights (All layers):\n")
    
            # Iterating over the Q-network layers
            for layer in self.online_network.layers:
                # Freezing the layers weights
                layer.trainable = False
                # Printing the layers 'trainable' parameter for sanity check
                print("{}: {}".format(layer.name, layer.trainable))
            
            print("\n=================================================================\n")
        
        # If the experiment_type is 3, apply the transfer learning from a pretrained Q-network
        if self.experiment_type == 3:
            # Apply transfer learning by replacing the pretrained Q-network output layer with a new one
            print("\n=================================================================\n")
            
            print("Removing the pretrained Q-network output layer...\n")

            self.online_network = tf.keras.Sequential(self.online_network.layers[:-2])

            print("Freezing the Q-network layers weights:\n")
    
            # Iterating over the q-network layers
            for layer in self.online_network.layers:
                # Freezing the layers weights
                layer.trainable = False
                # Printing the layers 'trainable' parameter for sanity check
                print("{}: {}".format(layer.name, layer.trainable))
            
            print("Adding a new output layer...\n")

            self.online_network.add(Dense(units=self.num_actions,
                                          activation='linear',
                                          name='Output'))
            
            print("Sanity check...\n")

            # Iterating over the model layers
            for layer in self.online_network.layers:
                # Printing the layers 'trainable' parameter for sanity check
                print("{}: {}".format(layer.name, layer.trainable))
            
            print("Recompiling the model...\n")
            
            # Configure the optimizer (Stochastic Gradient Descent)
            sgd = SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)

            # Recompile the model
            self.online_network.compile(loss='mean_squared_error',
                                        optimizer=sgd)

            print("\n=================================================================\n")
    

    def update_target(self):
        """
        Updates the target network weights to match the online network weights.

        Attributes:
            self.online_network: Neural network for online Q-value estimation.
            self.target_network: Neural network for target Q-value estimation.

        Returns:
            None: The updated weights are stored in the self.target_network attribute.
        """

        # Copy the weights from the online network to the target network. This procedure
        # ensures that the target network will reflect the most recent Q-value estimates
        self.target_network.set_weights(self.online_network.get_weights())
    

    def epsilon_greedy_policy(self, state):
        """
        Implements an epsilon-greedy policy for action selection.

        Args:
            state (numpy.ndarray): Current state representation.

        Returns:
            int: Selected action index.

        Attributes:
            self.total_steps (int): Total number of steps taken.
            self.epsilon (float): Exploration rate.
            self.num_actions (int): Number of possible actions.
            self.online_network (tf.keras.Model): Neural network for Q-value estimation.
            self.inference_time (list): Time taken for inference per step.
        """
        self.total_steps += 1
        
        # Check whether the probability to select a random action is less than or equal to epsilon
        if np.random.rand() <= self.epsilon:
            # If so, select a random action (exploration)
            return np.random.choice(self.num_actions)
        
        # Otherwise, select the action with the highest Q-value (exploitation)

        # Normalize the input state representation
        normalized_states = preprocessing.normalize(state)
        
        # Record the inference start time for performance monitoring
        starting_time = time.time()

        # Predict Q-values using the online network 
        q = self.online_network.predict(normalized_states)

        # Calculate the agent inference time (in seconds)
        self.inference_time.append(time.time() - starting_time)

        #print("\n=================================================================\n")
        #print("Action infered by the agent. Inference time in seconds: {0}".format(self.inference_time[-1]))
        #print("\n=================================================================\n")

        # Return the action matching the highest Q-value
        return np.argmax(q, axis=1).squeeze()
    

    def memorize_transition(self, s, a, r, s_prime, not_done):
        """
        Stores a transition (state, action, reward, next state, done) in the experience replay buffer.

        Args:
            s (numpy.ndarray): Current state.
            a (int): Action taken.
            r (float): Reward received.
            s_prime (numpy.ndarray): Next state.
            not_done (int): Flag indicating whether the episode is ongoing (1) or terminated (0).
        
        Returns:
            None: The transition is stored in the self.experience attribute.

        Attributes:
            self.episode_reward (float): Accumulated reward within the current episode.
            self.episode_length (int): Length of the current episode.
            self.train (bool): Flag indicating whether the agent is in training mode.
            self.epsilon (float): Exploration rate.
            self.epsilon_decay_steps (int): Number of steps for epsilon decay.
            self.epsilon_decay (float): Epsilon decay per step.
            self.epsilon_exponential_decay (float): Exponential decay factor for epsilon.
            self.episodes (int): Total number of episodes.
            self.rewards_history (list): History of episode rewards.
            self.steps_per_episode (list): Steps taken per episode.
            self.experience (collections.deque): Experience replay buffer.
        """

        # Check whether the episode has terminated
        if not_done == 0:
            # If the episode has terminated, update episode-related attributes
            self.episode_reward += r
            self.episode_length += 1
        # Otherwise, update exploration rate based on training progress
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

        # Update the episode counter
        self.episodes += 1
        # Update the reward history
        self.rewards_history.append(self.episode_reward)
        # Update the episode length
        self.steps_per_episode.append(self.episode_length)
        # Reset episode-specific attributes
        self.episode_reward, self.episode_length = 0, 0

        # Store the transition in the experience replay buffer
        self.experience.append((s, a, r, s_prime, not_done))
    

    def experience_replay(self):
        """
        Performs experience replay for training the Double Q-Network.

        Attributes:
            self.minimum_experience_memory (int): Minimum required experience in the replay buffer.
            self.experience (collections.deque): Experience replay buffer.
            self.batch_size (int): Batch size for training.
            self.online_network (tf.keras.Model): Neural network for online Q-value estimation.
            self.target_network (tf.keras.Model): Neural network for target Q-value estimation.
            self.gamma (float): Discount factor for future rewards.
            self.idx (Tensor): Tensor for batch indices.
            self.q_values (list): History of mean Q-values.
            self.losses (list): History of training losses.
            self.total_steps (int): Total number of steps taken.
            self.tau (int): Frequency for updating the target network.

        Returns:
            None: If the buffer does not have enough experience, return without updating the DQN.
                  Otherwise, train the online network with updated Q-values and update the target
                  network weights periodically.
        """

        # If the experience replay buffer does not meet the minimum requirement, no update occurs
        if self.minimum_experience_memory > len(self.experience):
            return

        # Sample a minibatch of transitions from the experience replay buffer
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        # Normalize the states and next states
        normalized_states = preprocessing.normalize(states)
        normalized_next_states = preprocessing.normalize(next_states)

        # Compute Q-values for next states using the online network
        next_q_values = self.online_network.predict_on_batch(normalized_next_states)
        # Select the best actions for next states
        best_actions = tf.argmax(next_q_values, axis=1)

        # Compute target Q-values using the target network
        next_q_values_target = self.target_network.predict_on_batch(normalized_next_states)
        
        # Gather target Q-values from the best actions considering the batch size 
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        # Computes target Q-values based on the Bellman equation
        targets = rewards + not_done * self.gamma * target_q_values

        # Record the mean Q-value for monitoring
        self.q_values.append(np.mean(targets))

        # Predict the Q-values for the sampled batch of transitions
        q_values = self.online_network.predict_on_batch(normalized_states)

        # Update the Q-values for the sampled actions
        q_values[self.idx, actions] = targets

        # Train the online network using the updated Q-values
        loss = self.online_network.train_on_batch(x=normalized_states, y=q_values)
        self.losses.append(loss)

        # Print training progress information
        print("\n========================================\n")
        print("learning from experience replay\n", end='\r')
        print("loss: {0} - mean q_value: {1}".format(loss, np.mean(targets)))
        print("\n========================================\n")

        # Update the target network weights periodically
        if self.total_steps % self.tau == 0:
            self.update_target()
    

    # Reset training-specific attributes
    def reset_metrics(self):
        self.losses = []
        self.q_values = []


    # Save the agent's state (both online and target networks)
    def save_agent(self, experiment_id):
        self.online_network.save('agent_models/online_network_{0}.h5'.format(experiment_id))
        self.target_network.save('agent_models/target_network_{0}.h5'.format(experiment_id))
