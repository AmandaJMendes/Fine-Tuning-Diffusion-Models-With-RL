import ImageReward as RM
import PIL
import PIL.Image
import torch
from aesthetics_predictor import AestheticsPredictorV1
from transformers import CLIPProcessor, logging, pipeline

DEFAULT_REWARD_PROMPT = (
    "a natural, high-quality portrait photograph of a person with realistic "
    "facial features, normal hair color, natural expression, and clean "
    "background"
)


@torch.no_grad()
def aesthetics_reward(pil_images: list[PIL.Image.Image]) -> list[float]:
    if not hasattr(aesthetics_reward, "_init"):
        aesthetics_model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
        aesthetics_reward.aesthetics_predictor = AestheticsPredictorV1.from_pretrained(
            aesthetics_model_id, device_map="cpu"
        )
        aesthetics_reward.aesthetics_processor = CLIPProcessor.from_pretrained(aesthetics_model_id)
        aesthetics_reward._init = True

    inputs = aesthetics_reward.aesthetics_processor(images=pil_images, return_tensors="pt")
    outputs = aesthetics_reward.aesthetics_predictor(**inputs)
    prediction = outputs.logits

    return prediction.squeeze().tolist()


@torch.no_grad()
def image_reward(pil_images: list[PIL.Image.Image], prompt: str = "") -> list[float]:
    if not hasattr(image_reward, "_init"):
        image_reward.image_reward_model = RM.load("ImageReward-v1.0", device="cpu")
        image_reward._init = True

    return image_reward.image_reward_model.score(prompt, pil_images)


@torch.no_grad()
def gender_reward(pil_images: list[PIL.Image.Image]) -> list[float]:
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
    return [next(p["score"] for p in preds if p["label"] == "male") for preds in classification]


def tensor_batch_to_pil_images(latents_batch: torch.Tensor) -> list[PIL.Image.Image]:
    # Move latents to CPU to save GPU memory
    latents_batch = latents_batch.detach().to("cpu", non_blocking=True)

    # Convert a tensor batch to list of PIL images
    image_processed = latents_batch.permute(0, 2, 3, 1)
    if image_processed.min().item() < 0:
        image_processed = (image_processed + 1.0) / 2.0

    image_processed = image_processed.clamp(0, 1).mul(255).byte().numpy()

    return [PIL.Image.fromarray(image_processed[i]) for i in range(image_processed.shape[0])]


@torch.no_grad()
def compute_reward_metrics(
    pil_images: list[PIL.Image.Image], prompt: str = DEFAULT_REWARD_PROMPT
) -> dict[str, torch.Tensor]:
    ir_person = image_reward(pil_images, prompt)
    sex_score = gender_reward(pil_images)
    aesthetics_score = aesthetics_reward(pil_images)

    sex_score = torch.as_tensor(sex_score, dtype=torch.float32)
    ir_person = torch.as_tensor(ir_person, dtype=torch.float32)
    aesthetics_score = torch.as_tensor(aesthetics_score, dtype=torch.float32)

    return {
        "ir_person": ir_person,
        "sex_score": sex_score,
        "aesthetics_score": aesthetics_score,
    }


@torch.no_grad()
def compute_total_reward(
    ir_person: torch.Tensor,
    sex_score: torch.Tensor,
    male_threshold: float = 0.8,
    gender_weight: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    sex_score_binary = (sex_score >= male_threshold).float()
    total_score = ir_person + gender_weight * sex_score_binary

    return total_score, sex_score_binary


@torch.no_grad()
def reward_function(
    latents_batch: torch.Tensor,
    prompt: str = DEFAULT_REWARD_PROMPT,
    male_threshold: float = 0.8,
    gender_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
