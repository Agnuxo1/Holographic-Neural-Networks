"""
Optical Raytracing Simulation for a Holographic Neural Network

This program simulates a holographic neural network based on optical raytracing.  It models the propagation of light between neurons in a 3D space and generates responses based on neuron activations. The goal is to explore the efficiency of this novel, physically-inspired neural network structure.

Author: Francisco Angulo de Lafuente
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
NUM_NEURONS = 100000  # Increased number of neurons for greater complexity
SPACE_SIZE = 100
RAY_INTENSITY = 0.005

# Initialize neuron positions in 3D space
def initialize_neurons(num_neurons, space_size):
    return np.random.rand(num_neurons, 3) * space_size

# Simulate light propagation between neurons
def simulate_light_propagation(neurons, input_text):
    activations = np.zeros(neurons.shape[0])
    # Simple simulation: associate words with random neurons
    hash_input = sum([ord(c) for c in input_text]) % neurons.shape[0]  # Corrected here
    for i in range(neurons.shape[0]):
        activations[i] = np.exp(-np.linalg.norm(neurons[i] - neurons[hash_input]) / SPACE_SIZE)
    return activations

# Generate a response using activations
def generate_response(activations):
    top_neurons = np.argsort(activations)[-5:]  # Take the 5 most active neurons
    # Generate a coherent response
    return f"The response is based on neurons {top_neurons} with activations {activations[top_neurons]}"

# Visualize neuron activation
def visualize_neurons(neurons, activations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors based on activation
    colors = activations / max(activations)
    ax.scatter(neurons[:, 0], neurons[:, 1], neurons[:, 2], c=colors, cmap='viridis', s=1)
    
    plt.title("Neuron Activation in 3D Space")
    plt.show()

# Main function for the holographic chat
def holographic_chat():
    neurons = initialize_neurons(NUM_NEURONS, SPACE_SIZE)
    
    print("Welcome to the raytracing-based holographic chat system.")
    print("Type 'exit' to end the chat.")
    
    while True:
        input_text = input("You: ")
        if input_text.lower() == 'exit':
            break
        
        print("Simulating light propagation between neurons...")
        time.sleep(1)  # Simulate processing time
        
        activations = simulate_light_propagation(neurons, input_text)
        print("Raytracing simulation complete.")
        
        response = generate_response(activations)
        print(f"Holographic Response: {response}")
        
        print("Visualizing activated neurons...")
        visualize_neurons(neurons, activations)

# Run the holographic chat
if __name__ == "__main__":
    holographic_chat()



