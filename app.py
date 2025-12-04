import torch
from flask import Flask, render_template, request, jsonify, send_file
from diffusers import ZImagePipeline
from diffusers.quantizers import PipelineQuantizationConfig
import threading
import time
import io
import base64
from PIL import Image

app = Flask(__name__)

# Global state for progress tracking
generation_state = {
    "status": "idle",  # idle, loading, generating, complete, error
    "progress": 0,
    "step": 0,
    "total_steps": 9,
    "message": "",
    "image": None
}

pipe = None
pipe_lock = threading.Lock()

def load_pipeline():
    global pipe
    generation_state["status"] = "loading"
    generation_state["message"] = "Loading model..."

    # 4-bit quantization config for transformer
    quantization_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True},
        components_to_quantize=["transformer"],
    )

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    generation_state["status"] = "idle"
    generation_state["message"] = "Model loaded"

def progress_callback(pipe, step, timestep, callback_kwargs):
    generation_state["step"] = step + 1
    generation_state["progress"] = int(((step + 1) / generation_state["total_steps"]) * 100)
    generation_state["message"] = f"Step {step + 1}/{generation_state['total_steps']}"
    return callback_kwargs

def generate_image_thread(prompt, width, height, steps, seed):
    global pipe
    try:
        with pipe_lock:
            if pipe is None:
                load_pipeline()

            generation_state["status"] = "generating"
            generation_state["progress"] = 0
            generation_state["step"] = 0
            generation_state["total_steps"] = steps
            generation_state["message"] = "Starting generation..."

            generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
                callback_on_step_end=progress_callback,
            ).images[0]

            # Convert to base64 for web display
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            generation_state["image"] = img_str
            generation_state["status"] = "complete"
            generation_state["progress"] = 100
            generation_state["message"] = "Generation complete!"

    except Exception as e:
        generation_state["status"] = "error"
        generation_state["message"] = str(e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "a beautiful landscape")
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 9))
    seed = int(data.get("seed", -1))

    if generation_state["status"] in ["generating", "loading"]:
        return jsonify({"error": "Generation already in progress"}), 400

    thread = threading.Thread(target=generate_image_thread, args=(prompt, width, height, steps, seed))
    thread.start()

    return jsonify({"status": "started"})

@app.route("/progress")
def progress():
    return jsonify(generation_state)

@app.route("/download")
def download():
    if generation_state["image"]:
        img_data = base64.b64decode(generation_state["image"])
        return send_file(io.BytesIO(img_data), mimetype="image/png", as_attachment=True, download_name="generated.png")
    return jsonify({"error": "No image available"}), 404

if __name__ == "__main__":
    print("Loading model on startup...")
    load_pipeline()
    print("Model loaded! Starting server...")
    app.run(host="0.0.0.0", port=5000, debug=False)
