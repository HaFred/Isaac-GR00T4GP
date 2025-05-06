## Rendering GR00T Omniverse data with Mujoco
As described in the GR00T paper, the major component of GR00T training data is the synthetic data constructed by Nvidia Nemo data flywheel. This flywheel calls both the omniverse isaac-sim env for physics simulation and NIM inference engine for AIGC data generation. Here we kickoff with the rendering to showcase the open Physics GR00T data released by Nvidia on huggingface.

For flywheel data curation, normally headless graphic cards servers are used. Therefore, we can run the Mojoco rendering as follows. A sample `gym-xarm` data is included in this folder for off-the-shelf visualization purpose.
```bash
MUJOCO_GL=egl python gym-xarm/example.py
```

If your Mojoco, opengl, egl are installed correctly. You will end up with the renderings below.

<video src="https://github.com/user-attachments/assets/0beb5971-8197-4c36-b38f-2bff204acb1a" autoplay muted loop playsinline></video>