import gymnasium as gym
import gym_xarm
import mediapy

env = gym.make(
    "gym_xarm/XarmLift-v0", 
    #    render_mode="human"  # comment this line to use headless mode
)
observation, info = env.reset()

images = []
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()
    if image is not None:
        images.append(image)
mediapy.write_video("out_vid_xarmlift.mp4", images, fps=60)
env.close()
