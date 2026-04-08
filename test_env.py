from env import StudentEnv

env = StudentEnv()
state = env.reset()

for i in range(10):
    state, reward, done = env.step(0)
    print("Step:", i)
    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)
    print("------")