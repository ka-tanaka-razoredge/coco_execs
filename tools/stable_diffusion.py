import base64
import os
import requests

STABLE_DIFFUSION_URL = "http://localhost:7860/sdapi/v1/txt2img"


def generate_image(prompt: str):

    payload = {
        "prompt": prompt,
        "steps": 20,
        "width": 512,
        "height": 512
    }

    res = requests.post(
        STABLE_DIFFUSION_URL,
        json=payload
    )

    image = res.json()["images"][0]

    os.makedirs("generated", exist_ok=True)

    filename = "generated/output.png"

    with open(filename, "wb") as f:
        f.write(base64.b64decode(image))

    return {
        "image": filename
    }
