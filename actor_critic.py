import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

#defining our actor critic class
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, tau=1.0):
        #calling our super constructor
        super(ActorCritic, self).__init__()
        #We are saving our gamma factor we gonna need that in our learning function.
        self.gamma = gamma
        self.tau = tau

        #We are defining our neural network it is 2d convolution,  i am gonna
        #pass in 0th element of our input, but because I believe
        #that as the number of channels or filter , kernal size, stride and padding
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        #the other layer are very similar so we gonna have 32 filters as inputs
        #and 32 filters as output, with 3 by 3, stride =2 qand padding =1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        #Now we dont know what the output of the fourth convolutional layer is,
        #so we have to use a function to figure it out
        #we conv_shaps that equals that mystry function which we're right in a moment
        conv_shape = self.calc_conv_output(input_dims)

        #next we have to do our gated recurrent unit.
        # and that will take whatever the output of our convolution layer is as input and then 256 hidden units
        self.gru = nn.GRUCell(conv_shape, 256)
        #and our policy pie will simply take 256 units of output and from our gru 
        #and convert it to a number of actions
        self.pi = nn.Linear(256, n_actions)
        #And similary to that value function, we want to take those 256 units as input and produce a single output
        self.v = nn.Linear(256, 1)

    #Dealing with that calc_conv_output function first
    def calc_conv_output(self, input_dims):
        #that just takes input dims as input, and we're just going to use,
        state = T.zeros(1, *input_dims) #we just gonna use an array of zeros as input, because the data itself doen't matter.
        #and if it not clear we need to add one dimension because the deep neuaral network is 
        #going to expect a batch of channel size by image size inputs so 
        # n /c/h/w. number  of batch size/ number of channels/height/ width
        #so we have one by starting with dimmension sort of be one by four by 42 by 42 as input. 
        
        #And we just pass deep down throgh our neural nework our convolutional network that is
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        dims = self.conv4(dims)
        #Than we want to return an integer as the product of the  size vector or tensor and 
        # that will tells us the exact number of units in our out put convolutional neural networks layers
        # And I think it turns out to be something like two hundred and eighty eight.
        return int(np.prod(dims.size()))

    #Handling feed forward function for our network 
    #this hx will take state and hidden state as input because we do have a gated recurrent layer in there
    def forward(self, state, hx):
        #THe reason I am going to use elu is  because it is consitent with ICM paper like to be
        #a little bit
        conv = F.elu(self.conv1(state))
        conv = F.elu(self.conv2(conv))
        conv = F.elu(self.conv3(conv))
        conv = F.elu(self.conv4(conv))

        #and  then we have to take that output and return a view,
        # a flattened view of it because the gated recurrent and return a flatterned view of it
        conv_state = conv.view((conv.size()[0], -1))
        #because gated recurrent expects a linear input
        
        #like so we can parse our convolutional state and that hidden state through our 
        #gated recurrent unit
        hx = self.gru(conv_state, (hx))
       
        # and then we can say pi or self.pi of that hidden state ajax and 
        #value function is likewise self taught
        pi = self.pi(hx)

        #and the value function is like wise self.v with that hidden state hx.
        v = self.v(hx)

        #now some other times and some other projects, I would have a choose action function for the agent
        # here i am just gonna handle all of this and these feed forward function.
        
        #So we're going to need a softmax activation on that policy output pie
        probs = T.softmax(pi, dim=1)
        #we gonna use that categorical distribution
        dist = Categorical(probs)
        #we are going to sample that distribution to get our action
        action = dist.sample()
        #Take the log probe of that action to get our actual log probe
        log_prob = dist.log_prob(action)
        #than we gonna return all those quantities the numpy array version of our action ,
        #because rember, action is a tensor, we want to  convert it to numpy and got our zeroth element that has to do with 
        # the dimensionality of the output it comes out of a tuple of an integer, so you just want to get the actual integer and not
        # tuple and than we have our value function v , log_prob, hx we need that hidden state for later calculations .
        return action.numpy()[0], v, log_prob, hx
############################This completes the basic functionality of actor crtic agent #######################

    #next we going to handle the way agent actually learns.
    #now we are calculating returns R from the paper 
    #this functions takes done flagm rewards and value list as input
    def calc_R(self, done, rewards, values):
        #the first thing we want to deal with is fact thar values is a list of tensor
        #rather than being a tensor by concatinating that list and than we going to squeeze that
        #because this comes out in shape 20 by one instead of 20 or whatever capital T is set to
        values = T.cat(values).squeeze()

        # the next thing we need to deal with is value of R at the teminal step we passed in
        if len(values.size()) == 1:  # batch of states
            #the below line follows defination of R from the paper
            R = values[-1]*(1-int(done))
        elif len(values.size()) == 0:  # single state
            R = values*(1-int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return,
                                dtype=T.float).reshape(values.size())
        return batch_return
    #We need to passs in intrinsic reward and default that to none
    def calc_cost(self, new_state, hx, done,
                  rewards, values, log_probs, intrinsic_reward=None):

        if intrinsic_reward is not None:
            rewards += intrinsic_reward.detach().numpy()

        returns = self.calc_R(done, rewards, values)
        #We want to deal with the question of what is the value of that new state.
        #So this is nessary because if you reflect back on the defination of Delta, we're going to have a 
        #We gonna hava term that goes as the value function for the state at time T plus one.
        #So  for the last state T, the Delta sub Capital T will still have that
        #term that says v subversive T plus one vivacity 
        #And so we need the value function for that T plus one state for the capital Tth time step
        #and we gonna get that by saying if the terminal flag is true then our next value will be a zero.
        #other wise it 'll be whatever our crtic says it is.
        next_v = T.zeros(1, 1) if done else self.forward(T.tensor(
                                        [new_state], dtype=T.float), hx)[1]
        #
        values.append(next_v.detach())
        #convert values to a pytorch tensor and again perform our squeeze operation
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)

        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps-t):
                temp = (self.gamma*self.tau)**k * delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)

        actor_loss = -(log_probs * gae).sum()
        # if single then values is rank 1 and returns rank 0
        # want to have same shape to avoid a warning
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)

        entropy_loss = (-log_probs * T.exp(log_probs)).sum()

        total_loss = actor_loss + critic_loss - 0.01 * entropy_loss
        return total_loss


