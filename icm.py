import torch as T
import torch.nn as nn
import torch.nn.functional as F

#Derived from the base nn.module
class ICM(nn.Module):
    #intialzation can take input dims, number of actions, lamda from the paper renamed as alpha because lamda is reserved keyword in python and beta
    def __init__(self, input_dims, n_actions=3, alpha=0.1, beta=0.2):
        super(ICM, self).__init__() #super constructer
        #And saving our hyperparameters
        self.alpha = alpha
        self.beta = beta
        
        #Now we have to handle the convolutional neural networks
        #That are going to serve the function of transforming our pixel
        #inputs into a feature representation.
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        #Recall that the output of a convolution neural network is our feature
        #representation, so I'm just going to call that phi keeping with the nomenclature in the paper
        self.phi = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        #Now we need use inverse model, which tells us, give successive states what action was taken.
        #and so it will have an input shape of 288 by 2 because it takes two of our feature representations
        #as input and then outputs 256 units 
        self.inverse = nn.Linear(288*2, 256)
        #and then we feed into another linear layer with outputs 
        # equal to the number to the number of actions
        #Now I  called it Pi Log, it's here because it's really going to give us the log it for our policy
        #We're going to be passing this to the cross entrophy loss function, which will take a softmax over these logits,
        #So we are not going to be doing any softmax activation here.
        self.pi_logits = nn.Linear(256, n_actions)

        #Next we have the forward model,which tells us, given state in action, what will the resulting state's
        self.dense1 = nn.Linear(288+1, 256)#And that will take 288+1 inputs because it's going to take a feature vector + an action which is represented by an integer. And i'll put 256 units.
        
        #And phi_hat_new just tells us what is the resulting state's represnting
        self.phi_hat_new = nn.Linear(256, 288)

        #finally, we have to send our network to a device 
        device = T.device('cpu')
        self.to(device)

    #Forward propagation operation, now this will take a state, new state and an action as input
    #forwar method to specify how data flows through the network
    def forward(self, state, new_state, action):
        #first we want to do is pass the state and new state through a convolutional layer to get phi and phi_new
        conv = F.elu(self.conv1(state))
        conv = F.elu(self.conv2(conv))
        conv = F.elu(self.conv3(conv))
        phi = self.phi(conv)

        conv_new = F.elu(self.conv1(new_state))
        conv_new = F.elu(self.conv2(conv_new))
        conv_new = F.elu(self.conv3(conv_new))
        phi_new = self.phi(conv_new)

        
        #Then we can handle the flattering of this output from our
        #convolutional neural network and reason we want to do that is because the outputs of our CNN is going to have the shape 
        # [T, 32, 3, 3] to [T, 288] so we are going to flattern this by using a view
        phi = phi.view(phi.size()[0], -1).to(T.float) # SO this will give us zeroth element of size, Which T and then minus one will just take
        #the product of 32 and 3 and 3

        # we have to do same thing for the phi_new
        phi_new = phi_new.view(phi_new.size()[0], -1).to(T.float)

        #Then we can take our inverse operation and get our pi_logit
        #to do that we need to concat phi and phi_new on the first dimension
        inverse = self.inverse(T.cat([phi, phi_new], dim=1))
        #And pass the outputs to pi_logits layer
        pi_logits = self.pi_logits(inverse)


        #Next we need to handle our forward model operation.

        
        #We gonna reshape our action it gonna go from # from [T] to [T, 1]
        action = action.reshape((action.size()[0], 1))
        #now we gonna concatinate those two vectors,  those two tenses along the first dimension 
        forward_input = T.cat([phi, action], dim=1)
        #and then pass it to the dense layer.
        dense = self.dense1(forward_input)
        #Than we can get our phi_hat new
        phi_hat_new = self.phi_hat_new(dense)
        
        #now we have to do is return the relevant quantities
        return phi_new, pi_logits, phi_hat_new

    #Finally we can handle the function to calculate a loss
    #And that also will take a state, newstate in action as input
    #Please make a note thes all states, new_states and actions are pural, Ostensibly there could be a single state, but
    #
    def calc_loss(self, states, new_states, actions):
        # don't need [] b/c these are lists of states
        # #Now keep in mind that  these are lists, so we don't need the brackets because these are lsits of states
        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        new_states = T.tensor(new_states, dtype=T.float)

        #than we can pass these through our forward operation
        phi_new, pi_logits, phi_hat_new = \
            self.forward(states, new_states, actions)
        #And calculate our losses, it's or inverse loss going to be the cross and to be lost between our pi_logits and actual action that the agent took
        inverse_loss = nn.CrossEntropyLoss()
        L_I = (1 - self.beta) * inverse_loss(pi_logits, actions.to(T.long))

        #forward loss is just the mean squared error loss of the phi_hat_new and phi_new
        forward_loss = nn.MSELoss()
        L_F = self.beta * forward_loss(phi_hat_new, phi_new)

        #And finally our intrensic reward  Now, One thing they didn't specify in the paper is the value for this hyper parameter ADA. Remember, it's ADA over two here, 
        #I've just set ADA to one, so it's one half. You can experiment with this i get decent result with an ADA of One half , so i stuck with it because there's a huge parameter space
        #to explore out there, and there isn't whole lot of guidance out there on what parameters to choose. So i stuck with one
        intrinsic_reward = self.alpha*0.5*((phi_hat_new-phi_new).pow(2)).mean(dim=1) #And we want to substract find new from phi_hat new and square it and take the mean along the first dimension. 
        #The above line of the code is probably the most critical line in the whole thing. So if you don't take the mean along the first dimension, this actually will not work now.
        #It won't work because it will take the mean across all elements and just give you a single number.
        #S0, what that means is you pass in 20 states and it gives you an intrensc reward for all 20 of those states and that doesn't make any sense, right?
        #It should give you an interensic reward for each state because each state has some curiosity associated with it. And so we definately have to apss in that dim equals one parameter.
        #It is easy mistake to make and pytorch wont actually compalin.
        #The dimensionalitu of everything works out fine because you,re just adding a single number to the reward later on in the A3C file so it doesn't get caught by any of the normal appartus and just goes right through and ruins.


        #So now we calculated our losses and reward and we can go ahead and return them
        return intrinsic_reward, L_I, L_F
