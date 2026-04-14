import ImageReward as RM
import numpy as np
import PIL
import torch
from aesthetics_predictor import AestheticsPredictorV1
from transformers import CLIPProcessor, logging, pipeline

DEFAULT_REWARD_PROMPT = (
    "a natural, high-quality portrait photograph of a person with realistic "
    "facial features, normal hair color, natural expression, and clean "
    "background"
)


def aesthetics_reward(pil_images):
    if not hasattr(aesthetics_reward, "_init"):
        aesthetics_model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
        aesthetics_reward.aesthetics_predictor = AestheticsPredictorV1.from_pretrained(
            aesthetics_model_id, device_map="cpu"
        )
        aesthetics_reward.aesthetics_processor = CLIPProcessor.from_pretrained(aesthetics_model_id)
        aesthetics_reward._init = True

    inputs = aesthetics_reward.aesthetics_processor(images=pil_images, return_tensors="pt")
    with torch.no_grad():
        outputs = aesthetics_reward.aesthetics_predictor(**inputs)
    prediction = outputs.logits

    return prediction.squeeze().tolist()


def image_reward(pil_images, prompt=""):
    if not hasattr(image_reward, "_init"):
        image_reward.image_reward_model = RM.load("ImageReward-v1.0", device="cpu")
        image_reward._init = True

    return image_reward.image_reward_model.score(prompt, pil_images)


def gender_reward(pil_images):
    if not hasattr(gender_reward, "_init"):
        logging.set_verbosity_error()
        gender_reward.pipe = pipeline(
            "image-classification",
            model="rizvandwiki/gender-classification",
            device="cpu",
            top_k=None,
        )
        gender_reward._init = True

    classification = gender_reward.pipe(pil_images)
    score_maps = [{pred["label"]: pred["score"] for pred in preds} for preds in classification]
    return [m["male"] for m in score_maps]


def tensor_batch_to_pil_images(latents_batch):
    # Move latents to CPU to save GPU memory
    latents_batch = latents_batch.to("cpu")

    # Convert a tensor batch to list of PIL images
    image_processed = latents_batch.permute(0, 2, 3, 1)
    if image_processed.min() < 0:  # [-1, 1]
        image_processed = (image_processed + 1.0) * 127.5
    else:  # [0, 1]
        image_processed = (image_processed) * 255

    image_processed = image_processed.numpy().astype(np.uint8)
    return [PIL.Image.fromarray(image_processed[i]) for i in range(image_processed.shape[0])]


def compute_reward_metrics(pil_images, prompt=DEFAULT_REWARD_PROMPT):
    ir_person = image_reward(pil_images, prompt)
    sex_score = gender_reward(pil_images)
    aesthetics_score = aesthetics_reward(pil_images)

    sex_score = torch.tensor(sex_score)
    ir_person = torch.tensor(ir_person)
    aesthetics_score = torch.tensor(aesthetics_score)

    return {
        "ir_person": ir_person,
        "sex_score": sex_score,
        "aesthetics_score": aesthetics_score,
    }


def compute_total_reward(ir_person, sex_score, male_threshold=0.8, gender_weight=2.0):
    sex_score_binary = (sex_score >= male_threshold).float()
    total_score = ir_person + gender_weight * sex_score_binary

    return total_score, sex_score_binary


def reward_function(
    latents_batch,
    prompt=DEFAULT_REWARD_PROMPT,
    male_threshold=0.8,
    gender_weight=2.0,
):
    images = tensor_batch_to_pil_images(latents_batch)
    metrics = compute_reward_metrics(images, prompt=prompt)
    total_score, sex_score_binary = compute_total_reward(
        metrics["ir_person"],
        metrics["sex_score"],
        male_threshold=male_threshold,
        gender_weight=gender_weight,
    )

    # Return all metrics for logging
    return total_score, {
        "ir_person": metrics["ir_person"],
        "sex_score": metrics["sex_score"],
        "sex_score_binary": sex_score_binary,
        "aesthetics_score": metrics["aesthetics_score"],
    }
