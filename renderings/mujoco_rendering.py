"""setup MUJOCO_GL=egl if you are also using headless env"""
import mujoco
import mediapy

model = mujoco.MjModel.from_xml_path("scene_sim/basic_scene.xml")
duration = 3.8
framerate = 60

data = mujoco.MjData(model)
frames = []
with mujoco.Renderer(model) as renderer:
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pix = renderer.render()
            frames.append(pix)

mediapy.write_video("out_vid.mp4", frames, fps=framerate)
