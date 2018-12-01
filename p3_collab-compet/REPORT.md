[//]: # (Image References)

[image1]: tennis_play.gif "Trained Agent"
[image2]: tennis_episode_scores.png "Score Episode Plot"

# Project 3: Collaboration and Competition

## Project Files
    1. Tennis.ipynb  -> Main training file to be run.
    2. agent.py      -> Code for ddpg, replay-buffer and noise
    3. multiagent.py -> Class to handle multiple ddpg agents
    4. network.py    -> Model neural networks for Actor and Critic


## Output model path
    1. checkpoint_actor.pth
    2. checkpoint_critic.pth


## How to load weights 
    ## Load ddpg agent 0
    ma.ddpg_agents[0].actor_local.load_state_dict(torch.load('checkpoint_actor.pth', map_location='cpu'))
    ma.ddpg_agents[0].critic_local.load_state_dict(torch.load('checkpoint_critic.pth', map_location='cpu'))

    ## Load ddpg agent 1
    ma.ddpg_agents[1].actor_local.load_state_dict(torch.load('checkpoint_actor.pth', map_location='cpu'))
    ma.ddpg_agents[1].critic_local.load_state_dict(torch.load('checkpoint_critic.pth', map_location='cpu'))


## MA-DDPG Algorithm

Google DeepMind has devised a solid algorithm for tackling the continuous action space problem. Building off the prior work on Deterministic Policy Gradients, they have produced a policy-gradient actor-critic algorithm called Deep Deterministic Policy Gradients (DDPG), that is off-policy and model-free, and that uses some of the deep learning tricks that were introduced along with Deep Q-Networks (hence the "deep"-ness of DDPG).

In this approach 2 networks are created for actor and 2 more networks are created separately for critic.
Multiple agents do not share same actor and critic here. Though, critic can be made common for all agents as in this particular problem action-space for each player is same.

Reference Paper [https://arxiv.org/pdf/1706.02275.pdf]


## Parameters and Observations
    BATCH_SIZE = 256
    HIDDEN_LAYERS = (256,128)
    UPDATE_EVERY = 4
    
Smaller network less than 128 hidden units were not learning properly. Also, with batch size <256 learning was slower.
When updated every single step, learning was neglegible also learning every 8th or more step scores were oscillating more.
Update every 4th step was kind of optimal number.


## Episode vs Scores plot
![Tennis Episode Scores][image2]



### Average score of 1.94 was achieved after 500 iterations.
### Moving Average score of >0.5 was achieved at 100th iteration easily.
### Scores started oscillating from 1.9 to 2.0 from iteration 100 and did not improve further.



## Trained Agnets Playing
![Agents Playing][image1]



## Further Improvements

1. As mentioned above same critic network can be used among agents
2. Tried solving "optional" soccer-problem with same strategy but clearly not working. Created separate multiagents for goalkeeper and striker as both had separate action-space but training did not converge.
3. Need to go through some papers related to "Multiagent", "Co-operative" and "Competitive" environment as simple strategies like maddpg are not useful for team-game like soccer.


