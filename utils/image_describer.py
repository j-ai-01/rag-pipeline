import base64
from pathlib import Path
from ollama import Client
from config import OLLAMA_BASE_URL, VISION_MODEL


def describe_image(path: Path) -> str:
    client = Client(host=OLLAMA_BASE_URL)
    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    response = client.generate(
        model=VISION_MODEL,
        prompt=(
            "Describe this image in detail. "
            "Include all visible text, objects, colors, charts, diagrams, "
            "and any other relevant information you can see."
        ),
        images=[image_b64],
    )
    return response["response"]
