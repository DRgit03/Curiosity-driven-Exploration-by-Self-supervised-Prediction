# Curiosity-driven-Exploration-by-Self-supervised-Prediction
Self supervised predictions of environment dynamics  ->Only incentive learning dynamics realted to actions ->Agent learns with sparse or absent in extrensic rewards 

Overview of ICM paper:
=======================
We have to address the issue of learning environment with sparse rewards. This is an important topic because  reward sparsity is going to scale with how closely a system mimics real life and real life rewards are few and  far between.
If reinforcement learning is going to be route to general intelligence it must deal with this fact.
->Here we need to utilize a sort of self prediction about the future state of system and intrensic reward is calculated in terms of how far often a agent is from being able to predict state.
->Transitions in the system to predict state tansitions in the system in terms  of reduced dimensional feature representation of the space.
->In Other words some space that is related to the pixel of an image but is not actual pixel in the image.
-> The agent is also incentivized ti learn how actions affect the system again in temrs of the time evolution of the reduced dimensional feature represntation.
->The result is that they're able to show the agent not only learns in presence of sprase rewards, but also in the presence of no extresnsic rewards at all.
->It also shows a sort of transfer learning ability where the agent is able to learn in one environment and tested in another related yet different environment.

ICM in Nutshell:
=================
->Self supervised predictions of environment dynamics 
->Only incentive learning dynamics realted to actions
->Agent learns with sparse or absent in extrensic rewards
ICM: Curiosity Driven Exploration
======================
-> We gonna have two neural networks one for the agent and one for the ICM.
->Training is going to be facilitated by taking the sum of intrensic and extrensic reward
->The policy is represented by a network.
-> We going to using  the sum going to use the sum og intrensic and extrensic rewards at each steps, they are going to be using ICM wih A3C
Key points:
->Explore without raw pixels.
->Encode pixel inputs to abstract feature vector
->Modules for forward and inverse environement
->Loss function only decode changes due to agent's actions.
->Add curiosity loss to extrensic loss and minimize total loss.

Experimemtal setups and coding our ICM module
=============================================
4 convolutional layers ->Same as A3C.
->Inverse: Concat two feature vectors and pass through linear layer with 256 units, followed by output with n_actions
->Forward: Concat feature vector and action and pass through 2 linear layers with 256 and 288 units
->Lamda 0.1, beta 0.2, learning rate 10^-4
->Modify worker, parallel_env, main to accept global icm optimizer.
->Modify memory to store states, newstates and  actions


Key points:
Dense, sparse rewards compared.
-> pixel learning ok in dense / sparse but fails in very sparse, Full ICM best in all cases
->Noise didn't hurt ICM + A3C but ICM + pixcels suffered.
->Training with no reward lead to good exploration.
->Training with no rewards did good in subsequent mario levels

Built on modules for other algorithms 
->encode pixels inputs to abstract feature vector

Network for forward and inverse module.
->Forward: Current state + action of next state feature vector.
->Inverse:Current state + next state ->action taken

curiosity reward: error between feature state vectors
->One new class for ICM (Inverse Control Model)module
->Test in Maze environment.


Code structure:
===============
main.py: call the code to fire up our threads.
parellel_env.py: start and join threads, global agent / optim.
worker.py->play episodes and learn  for each thread.
memory.py  rudimantary batch memory for agents.
actor_critic.py: network, feed forward, loss function.
utils.py: plot learning curve to visualize performance
wrappers.py modify the open ai gym environment.


