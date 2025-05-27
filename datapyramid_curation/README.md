## Neural Trajectories Generation
With Video Gen models like [Wan2.1](predict_i2v.py) and [Cosmos-transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1), we can augment more trajectory data entries in an NN way.

Here we illustrate a sample case about how this can be done, the GT data and input frames are coming from Omniverse simulation.

| Initial Input Frame | Ending Input Frame | Groundtruth <br> (from Omniverse) | Pretrained Wan2.1-FLF2V-14B | Pretrained Wan2.1-Fun-V1.1-1.3B-InP | Finetuned Wan2.1-Fun-V1.1-1.3B-InP |
|---|---|:---:|---|---|---|
|   |   |   |   |   |   |