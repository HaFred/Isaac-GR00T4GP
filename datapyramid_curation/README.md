## Neural Trajectories Generation
With Video Gen models like [Wan2.1](predict_i2v.py) and [Cosmos-transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1), we can augment more trajectory data entries in an NN way.

Here we illustrate a sample case about how this can be done, the GT data and input frames are coming from Omniverse simulation.

<table border="0" style="width: 100%; text-align: left; table-layout:fixed; margin-top: 20px;">
    <tr>
        <th>Initial Input Frame</th>
        <th>Ending Input Frame</th>
        <th>Groundtruth <br>(from Omniverse)</th>
        <th>Pretrained Wan2.1-FLF2V-14B</th>
        <th>Pretrained Wan2.1-Fun-V1.1-1.3B-InP</th>
        <th>Finetuned Wan2.1-Fun-V1.1-1.3B-InP</th>
    </tr>
      <td>
          <image src="https://github.com/user-attachments/assets/105a35cc-5844-4f2d-8029-b0b60e26fbb1" width="100%" controls autoplay loop></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/6691075c-6fc8-49fb-b3fc-9467e0c8f74d" width="100%" controls autoplay loop></image>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1e3bda56-3b6b-4b92-9a90-3802d23a73f0" width="100%" controls autoplay loop></video>
     </td>
       <td>
          <video src="https://github.com/user-attachments/assets/702ae91f-a138-4847-8229-5550d447f1ee" width="100%" controls autoplay loop></video>
     </td>
    <td>
          <video src="https://github.com/user-attachments/assets/438905b7-e11b-45a7-87f5-d98b4e2fd2fd" width="100%" controls autoplay loop></video>
     </td>
    <td>
          <video src="https://github.com/user-attachments/assets/34778cd5-fb5b-4a58-bb14-1e0c67784e6a" width="100%" controls autoplay loop></video>
     </td>
  <tr>
</table>
