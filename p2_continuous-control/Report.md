# CONTINUOUS CONTROL - PROJECT OPTION 1 SOLUTION

### Code Files 

1. Ipython Files

    a. **Continuous_Control - Problem 1 - Single Agent.ipynb** => Project-Option-1 solved with one agnet only (For submission).

    b. **Continuous_Control - Problem 2 - Multiple Agent.ipynb** => Tried to solve project-option-2 with 20 agnets (Testing).

2. Agent Files

    a. **ddpg_agent.py** => Deep Q Agent code very similar to **ddpg** project in nanodegree

3. Model Files

    a. **model.py** => Deep Q Network code very similar to **ddpg** project in nanodegree. 
        But smaller network did not converge, so tried duel/triple layer dense network for both actor and critic.

4. Model Checkpoint Files

    a. **checkpoint_actor_p1.pth** => Project-option-1 actor model file

    b. **checkpoint_critic_p1.pth** => Project-option-1 critic model file


### Learning Algorithm

**Deep Deterministic Policy Gradient (DDPG)**

Although DQN achieved huge success in higher dimensional problem, such as the Atari game, the action space is still discrete. However, many tasks of interest, especially physical control tasks, the action space is continuous. If you discretize the action space too finely, you wind up having an action space that is too large. For instance, assume the degree of free random system is 10. For each of the degree, you divide the space into 4 parts. You wind up having 4¹⁰ =1048576 actions. It is also extremely hard to converge for such a large action space.

DDPG relies on the actor-critic architecture with two eponymous elements, actor and critic. An actor is used to tune the parameter 𝜽 for the policy function, i.e. decide the best action for a specific state.

A critic is used for evaluating the policy function estimated by the actor according to the temporal difference (TD) error.

![DDPG Algorithm](ddpg_algorithm.png)


#### Best Parameters found through experiment 
n_episodes=2000

eps_start=1.0

eps_end=0.1

eps_decay=0.995


### Plot of Rewards

#### DQN Scores Plot

![DQN Scores](navigation_dqn_score_plot.png)

Environment solved in 642 episodes

Scores => mean - 8.174, deviation - 5.525

Window Avg Scores => mean - 7.166, deviation - 4.435

#### DDQN Scores Plot

![DDQN Scores](navigation_ddqn_score_plot.png)

Environment solved in 486 episodes

Scores => mean - 6.872, deviation - 5.244

Window Avg Scores => mean - 5.519, deviation - 4.220


### Ideas for Future Work

1. Using different kind of network (RNN, LSTM, CNN) apart from default network given in dqn project.

2. Using keras in the backend instead of pytorch

3. Try different kind of Deep Q network strategy

4. Still trying out pixel code and not completely ready as it is taking time to run network. Looking at this code for reference and trying out --> https://github.com/gtg162y/DRLND/tree/master/P1_Navigation/visual_pixels

