from env import StudentEnv
from agent import Agent

env = StudentEnv()

state_size = 4
action_size = 5

agent = Agent(state_size, action_size)

episodes = 300

for e in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.act(state)

        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {e}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            break

# SAVE MODEL
agent.save()