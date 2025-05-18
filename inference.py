from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.transforms import GR00TTransform, DEFAULT_SYSTEM_MESSAGE
from gr00t.model.backbone.eagle2_hg_model.inference_eagle_repo import EagleProcessor

# # 1. Load the modality config and transforms, or use above
# modality_config = ComposedModalityConfig(...)
# transforms = ComposedModalityTransform(...)
DATASET_PATH = "/home/fredhong/test/demo_data/robot_sim.PickNPlace"
MODEL_PATH = "/data0/gr00t"

# get the data config
data_config = DATA_CONFIG_MAP["gr1_arms_only"]

# get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

# 2. Load the dataset
dataset = LeRobotSingleDataset(
    dataset_path="../test/demo_data/robot_sim.PickNPlace",
    modality_configs=modality_config,
    transforms=None,  # we can choose to not apply any transforms
    embodiment_tag=EmbodimentTag.GR1, # the embodiment to use
)
# 3. Load pre-trained model
policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.GR1,
    device="cuda"
)

# 4. Run inference
action_chunk = policy.get_action(dataset[0])

# 5. Check how the tokenizer is encoding the text prompt
batch = {
    "language": "pick the pear from the counter and place it in the plate"
}
trans = GR00TTransform(
    state_horizon=len([0]),
    action_horizon=len(list(range(16))),
    max_state_dim=64,
    max_action_dim=32,
)
vlm_processor = EagleProcessor()
prompt = [
    {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
    {
        "role": "user",
        "content": batch["language"],
    },
]
vlm_inputs = vlm_processor.prepare_input({"prompt": prompt})
print(vlm_inputs['input_ids'])

# 6. Use the vlm and system 1 to generate actions
print("\n****************\n")
print(dataset[0])
print("\n****************\n")
print(action_chunk)

# 7. Use the policy's tokenizer to decode a chat temp,
# then we can futher put sth like "wait..." to do the 
# tts and potentially imporve the action generation
