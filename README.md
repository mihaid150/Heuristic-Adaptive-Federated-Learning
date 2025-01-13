# Heuristic-Adaptive Federated Learning Framework
This project aims to create a comprehensive solution for deploying and simulating a Federated Learning network which comes with a novel approach to optimize the model learning process. It is consisting of two main heuristical algorithms, Simulated Annealing and Genetic Algorithm. On one side we create a tailored solution for balancing the exploration/exploation search for performant models while on the training process we are searching at the level of each cluster of clients, training hyper-parameters to ensure a reliable optimization process.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [APIs](#apis)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
Federated Learning is a distributed machine learning process in which the learning part it is decoupled from a deep large model into multiple smaller collaborating models. Its distributed nature imposes a client-server approach in which there would be several clients holding the private data and one or more servers which would aggregate the resulting models and share them back again for retraining. 

In our solution we are using a hierarchical topology of the network splitted in 3 layers, Cloud, Fog and Edge and each type of node would have at least one of these roles. Beside this nodes organization, the training process it is designed to explore new solutions(models) or exploit the current one based on the system's temperature, in addition with a fine-tuning process for searching optimal training parameters that provide the right path for model's performance convergence.   

## Features
 - Aggregation of models in 2 stages, at the level of fog and at the level of cloud nodes.
 - Supports LSTM-based Keras models.
 - Provides real-time traffic and performance statistics generation.
 - Implementation of a cooling scheduler to act as a Simulating Annealer in the fog and cloud layers and a genetic engine to be used by fog nodes during hyper-parameters optimization.
 - Modular design for extensibility.
 - Caching model files, training statistics and genetic engine configuration for a training flexibility.

# Architecture
The system follows a cloud-fog-edge hierarchy topology:
1. **Cloud Node** Aggregates models, process performance and other statistics data, orchestrates fog and edge nodes, runs a simulated annealer for providing to fog nodes a temperature-based genetic offsprings evolution.
2. **Fog Node** Intermediate layer for aggregating edge retrained model, orchestrates the evaluation and training edge nodes, runs the simulated annealer for model selection, runs the genetic engine for hyper-parameters optimization.
3. **Edge Node** Generate and (re)trains the local edge models based on the privately stored data.

<img src="./images/architecture-diagram.png" alt="Architecture Diagram" width="70%">
