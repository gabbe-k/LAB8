import gym 
import network_ttt
from network_ttt import Agent
from gym_ttt import TicTacToeEnv
import numpy as np
import matplotlib.pyplot as plt
import torch as t

def plot_learning_curve(x, scores, epsilons, filename, ylabel="Score"):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    n = len(scores)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel(ylabel, color="C1")  
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)

    

def main():

    n = 3 
    env = TicTacToeEnv(n = n)

    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size=64, n_actions=n*n, eps_end=0.01, input_dims=[n*n], lr=0.001)
    scores, eps_history, losses = [], [], []
    n_games = 10000
    for i in range(n_games):
        score = 0 
        done = False
        observation = env.reset()


        while not done:
            #print("episode: ", i)

            #print("observation:")
            #print(observation.reshape(3,3))

            # discrete observation between 0 and 8
            action = agent.choose_action(observation)

            #print("action: ", action)

            observation_ , reward, done, info = env.step(action, i)

            #print("observation_: ")
            #print(observation_.reshape(3,3))

            #print("reward: ", reward)

            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            loss = agent.learn()
            #set current state to new state
            observation = observation_

        


        scores.append(score)

        if not loss:
            loss = 0

        losses.append(loss)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        if i % 500 == 0:
            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon, 'loss %.2f' % loss)

    x = [i+1 for i in range(n_games)]
    filename = 'moon_landing_fake.png'
    filename2 = 'bush.png'

    plot_learning_curve(x, scores, eps_history, filename)
    plot_learning_curve(x, losses, eps_history, filename2, ylabel="Loss")

    #Save the model
    t.save(agent.Q_eval.state_dict(), 'ttt_model.pt')


if __name__ == '__main__':
    main()