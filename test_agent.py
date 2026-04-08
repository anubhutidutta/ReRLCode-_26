from env import StudentEnv
from agent import Agent

env = StudentEnv()

agent = Agent(4, 5)
agent.load()

state = env.reset()

while True:
    action = agent.act(state)
    state, reward, done = env.step(action)

    print("State:", state, "Reward:", reward)

    if done:
        print("Game Over")
        break