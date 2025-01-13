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

## Architecture
The system follows a cloud-fog-edge hierarchy topology:
1. **Cloud Node** Aggregates models, process performance and other statistics data, orchestrates fog and edge nodes, runs a simulated annealer for providing to fog nodes a temperature-based genetic offsprings evolution.
2. **Fog Node** Intermediate layer for aggregating edge retrained model, orchestrates the evaluation and training edge nodes, runs the simulated annealer for model selection, runs the genetic engine for hyper-parameters optimization.
3. **Edge Node** Generate and (re)trains the local edge models based on the privately stored data.

<img src="./images/architecture-diagram.png" alt="Architecture Diagram" width="70%">

## Technologies Used
- **Java (Spring Boot)**: Backend framework.
- **Python (Tensorflow):** For model creation, handle and aggregation scripts together with heuristic's components.
- **Docker**: Deploy and containerization.
- **ReactJS**: Frontend managing platform.
- **WebSocket**: Real-time communication between FE and BE.

## Setup and installation
### 1. Clone the repository
```
git clone https://github.com/mihaid150/Heuristic-Adaptive-Federated-Learning
```
### 2. Network configuration and prerequisites
As node devices we used Raspberry Pi 5 and Pi 4 (RPI5/RPI4) boards for the edge layer and some InterCore I3 Workstations for fog and cloud layer. The setup could be done fine also on a network with only RPIs boards or only Intel workstations. Next i will present how to install Ubuntu Server on those two types of nodes.

#### 2.1 Installing Ubuntu Server on a Raspberry Pi Node
a. **Download the Rasbperry Pi Imager**:
   - Visit the [official Raspberry Pi page](https://www.raspberrypi.com/software/) and download the latest version of the Raspberry Pi Imager application whether it is for x86 or macOS.
b. **Raspberry Pi Image Configuration**:
   - Insert the board MicroSD card(I recommend to be at least 64 GB for no futher worries) into the working desktop and open the Raspberry Pi Image app. From the menu, select the storage representing the MicroSD Card, the operating system that we will install (Choose OS -> Other general-purpose OS -> Ubuntu) should be an LST server version for stability reasons so I have chose Ubuntu Server 24.04.1 LTS (64-bit). From the Raspberry Pi Device field I have selected Raspberry Pi 5 (4) depending on node, I recommend to be at least RPI4 with at least 4 GB RAM memory.
   - After these 3 fields, you can press NEXT and you will be asked if you want some additional customizations and press on EDIT SETTINGS. In the new opened window, set the hostname, the username(I have followed the format *pinodeX* where X can be an index from the ordering rule, e.g., *pinode5*) and the password, configure also the wireless connection with the SSID and password, I recommend to choose a stable network for connection. Then press to SAVE and press YES 2 times and the MicroSD card formatting should begin. When its ready you can plug the MicroSD card into the RPI.
     
<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="./images/ubuntu_server_version.png" alt="Ubuntu Server Version" width="40%">
  <img src="./images/rpi_imager_1.png" alt="Raspberry Pi Imager (1)" width="40%">
</div>
<img src="./images/rpi_imager_2.png" alt="Raspberry Pi Imager (2)" width="30%">
