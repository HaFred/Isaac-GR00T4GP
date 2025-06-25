from typing import Optional, Tuple
from safetensors.torch import load_file
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from datetime import datetime

"""
Getting with below dumps. This is to see how should correctly patchify the blocks for lavida diffusion-LM backbone instead the eagle2 backbone.

>> from safetensors.torch import save_file
>> save_file({"eagle": images}, "eagles_correctlyproc_imgs.safetensors")

from:

1. `backbone.llada.llava_arch.prepare_inputs_labels_for_multimodal()` 

2. Lavida, /home/fredhong/diffusion_lm/LaViDa/llava/model/llava_arch.py
"""
now = datetime.now()
datetime_string = now.strftime("%Y-%m-%d-%H:%M:%S")

def show_img(imgs: Optional[Tuple], naming_of_input="eagle"):
    # if not isinstance(imgs, (list, tuple)):
    #     imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # plt.imsave("eagle_img.png", np.asarray())
    # fig.savefig(f'{naming_of_input}_tokenized_lavida_img.png')
    fig.savefig(f"{datetime_string}.png")

    return datetime_string

def show_img_int(img: np.array):
    assert len(img.shape) == 3
    if img.shape[0] == 3: # it's c h w, permute it
        img = np.transpose(img, (1, 2, 0))
    f, axs = plt.subplots(ncols=1, squeeze=False)
    img = F.to_pil_image(img)
    axs[0, 0].imshow(np.asarray(img))
    axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    f.savefig('step_datarframe.png')


if __name__ == "__main__":
    # a = load_file("/home/fredhong/gr00t4gp-origin/gr00t/eagle_on_llava_images.safetensors")
    # a = load_file("/home/fredhong/gr00t4gp-origin/gr00t/eagle_correct.safetensors")
    a = load_file("/home/fredhong/gr00t4gp-origin/gr00t/2noegaleproc_dogeimg.safetensors")
    in_tensor = a['eagle'].to(torch.float32)
    # a = load_file("/home/fredhong/diffusion_lm/LaViDa/lavida_imgs.safetensors")
    # in_tensor = a['lavida'].to(torch.float32)

    print(in_tensor.shape)

    # vis a
    plt.rcParams["savefig.bbox"] = 'tight'

    # import pdb; pdb.set_trace()
    in_tensor = torch.unbind(in_tensor)
    # print(in_tensor)
    # print(len(in_tensor))

    # show(in_tensor, "lavida")
    show_img(in_tensor, "2noegaleproc_dogeimg")