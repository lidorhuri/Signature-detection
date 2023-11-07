# Project Overview: Signature Authentication System

# Introduction
This project introduces a signature authentication system that leverages a React-Native application, a NodeJS server, and multiple Python services to create a robust signature verification process. The system is designed to learn from the user's unique signature characteristics during registration and use this information to verify identity during login attempts.

# System Workflow
# Registration
During the registration phase, the user is required to provide 10 instances of their signature through the React-Native application. These samples are sent to the server, which coordinates the learning process across several Python-based services.

# Login
When logging in, the user provides a signature which is evaluated by the system. If the signature is not recognized, a message indicating the failure stage is returned. A successful match results in a success message.

# Authentication Steps
# Step 1: Raising Hands Verification
The system analyzes the signature to verify the number of times the user raises their hand. This step ensures that the hand-lifting pattern during login matches the pattern observed during training.

# Step 2: Direction Check
The system assesses the directionality of the signature strokes. It distinguishes between signatures based on the direction in which lines are drawn (e.g., top-to-bottom vs. bottom-to-top), adding a layer of complexity to the authentication process.

# Step 3: SVM Model Learning
The system employs a Support Vector Machine (SVM) model to learn the user's signature style from the 10 samples provided. It creates a template that is used to check if at least 75% of the login signature fits within the learned template.
![Figure_1](https://github.com/lidorhuri/Signature-detection/assets/123116810/3f54d072-f804-434a-804b-e4d0f169e366)


# Step 4: DC-GAN Model Learning
Using CUDA for accelerated processing, the system trains a Deep Convolutional Generative Adversarial Network (DC-GAN) model to generate signature forgeries. Once the model can successfully forge signatures, it uses this ability to verify the authenticity of login signatures against potential forgeries.

![image](https://github.com/lidorhuri/Signature-detection/assets/123116810/191f244d-d49b-4b2e-ac66-e02b67365ac0) ![image](https://github.com/lidorhuri/Signature-detection/assets/123116810/a4713365-9bdf-4c70-bb33-ea7617275b0e)


# Accuracy Ranges
The system's accuracy in identifying signatures varies, with a range of 75% to 93%. This variation is due to the unique nature of each signature and the different learning outcomes of the models.

# Running the Project
# Client-Side Setup
1. Utilize a temporary server such as NGROK and set the server link using the NGROK_LINK variable in App.tsx.
2. Activate the client using the command expo-cli start --tunnel. Scan the provided code with the Expo application.
   
# Server-Side Setup
1. Run the NodeJS server and obtain the server link to update the NGROK_LINK in the client application.
Python Services
2. The Python services will run smoothly if the path in the Constant file is correctly set.

# * It is recommended to install and use CUDA
