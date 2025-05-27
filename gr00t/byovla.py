import json

# Imports
from PIL import Image, ImageFilter
import base64

# For hf and kimi
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def perturb_gaussian_noise(image, mask, std=0.25):
    """
    Input:
    image: numpy array of shape (H, W, 3)
    mask: numpy array of shape (H, W) - where to add noise

    Output:
    noised_image: numpy array of shape (H, W, 3)
    """

    # Convert the image to a float32 type
    image = image.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0

    # Define the Gaussian noise parameters
    mean = 0
    std_dev = std
    gaussian_noise = np.random.normal(mean, std_dev, image.shape)

    # Add the Gaussian noise to the image
    gaussian_noise[:, :, 0] = np.where(mask > 0, gaussian_noise[:, :, 0], 0)
    gaussian_noise[:, :, 1] = np.where(mask > 0, gaussian_noise[:, :, 1], 0)
    gaussian_noise[:, :, 2] = np.where(mask > 0, gaussian_noise[:, :, 2], 0)
    noisy_image = image + gaussian_noise
    # Clip the values to [0, 1] and convert back to [0, 255]
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)

    return noisy_image


def perturb_gaussian_blur(image, mask, kernel_size=25):
    # Apply Gaussian blur to the whole image
    # The mask must be 0-255, not 0-1

    mask = mask * 255
    mask = mask.astype(np.uint8)

    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=kernel_size))

    # Composite the original image with the blurred image using the mask
    blurred_region = Image.composite(blurred_image, image, mask)
    blurred_region = np.asarray(blurred_region)
    return blurred_region


def kimi(image_path, lang_ins):
    # Read in examples for few-shot learning
    # testtime_image = encode_image(img_path)
    testtime_image = Image.open(image_path)

    # Create context and run with kimi
    payload = {
        "messages": [
            { 
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant helping a robot determine what objects in the image are relevant for completing its task.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You will be shown some text and images.",
                    },
                    {"type": "text", "text": '["obj1", "obj2", "obj3"]'},
                    {"type": "text", "text": '["background1", "background2"]'},
                    {
                        "type": "text",
                        "text": "The robotic arm in the image is given the following task: "
                        + lang_ins
                        + ". Provide a list of objects in the image that are not relevant for completing the task, called 'not_relevant_objects'. Then provide a list of backgrounds in the image that are not relevant for completing the task, called 'not_relevant_backgrounds'. Give your answer in the form of two different lists with one or two words per object. Respond in JSON file format only.",
                    },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{testtime_image}"
                    #     },
                    # },
                    {
                        "type": "image",
                        "image": image_path
                    }
                ],
            },
        ],
        "max_tokens": 300,
    }

    model_path = "moonshotai/Kimi-VL-A3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )

    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": lang_ins}]}
    ]
    text = processor.apply_chat_template(
        payload["messages"], 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    inputs = processor(images=testtime_image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=payload["max_tokens"])
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # print(response)
    return response


def vlm_refine_output(response):
    """
    Input:
    response: json file from GPT4 containing list of objects and backgrounds in image not relevant to task

    Output:
    not_relevant_list: list of objects not relevant to task
    """
    # Parse the JSON
    parsed_response = json.loads(response)

    # # Extract information
    # all_objects = parsed_response["choices"][0]["message"]["content"]

    # # Refine the JSON string
    # all_objects_refine = all_objects.replace("json", "")
    # all_objects_refine = all_objects_refine.replace("```", "")

    # parsed_all_objects = json.loads(all_objects_refine)

    # Extract relevant and not relevant objects into separate lists
    not_relevant_objects_list = parsed_response["not_relevant_objects"]
    not_relevant_backgrounds_list = parsed_response["not_relevant_backgrounds"]

    return not_relevant_objects_list, not_relevant_backgrounds_list


if __name__ == "__main__":
    # inputs
    language_instruction = "place the carrot on plate"
    img = "test_byovla.png"

    # call vlm
    vlm_output = kimi(img, language_instruction)
    print(f"init vlm output:\n{vlm_output}\n")
    no, nb = vlm_refine_output(vlm_output)
    print(
        f"after refinement, not relevant object:\n{no}\n" +
        f"not relevant background:\n{nb}"
    )