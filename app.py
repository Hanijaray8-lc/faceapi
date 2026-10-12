# pyright: reportMissingImports=false
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import io, base64, os

try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore

device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
app = Flask(__name__)

# Enable CORS for frontend integration
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response

print(f"Loading VASTRA on {device}...")
try:
    from model.pipeline import CatVTONPipeline  # type: ignore
    from model.cloth_masker import AutoMasker  # type: ignore
    automasker = AutoMasker(device=device)
    pipeline = CatVTONPipeline(
        base_ckpt="runwayml/stable-diffusion-inpainting",
        attn_ckpt="zhengchong/CatVTON",
        attn_ckpt_version="mix",
        weight_dtype=(torch.float16 if device == "cuda" else torch.float32) if torch else None,
        device=device
    )
    print("VASTRA loaded successfully!")
except Exception as e:
    print(f"Model load: {e}")
    automasker = None
    pipeline = None

def decode(b64, target_size=(768, 1024)):
    if ',' in b64:
        b64 = b64.split(',')[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return img.resize(target_size, Image.LANCZOS)

def encode(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "engine": "VASTRA HD 4K - Face Same + Dress Fit",
        "resolution": "768x1024 HD",
        "steps": 40,
        "quality": "high"
    })

@app.route("/tryon", methods=["POST", "OPTIONS"])
def tryon():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if pipeline is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    data = request.get_json(force=True)
    p_b64 = data.get("person_image") or data.get("personImage")
    c_b64 = data.get("cloth_image") or data.get("garmentImage")

    if not p_b64 or not c_b64:
        return jsonify({"status": "error", "message": "person and cloth images required"}), 400

    person_orig = decode(p_b64, (768, 1024))
    person = person_orig.copy()
    cloth = decode(c_b64, (768, 1024))

    # HD Mask - Dress area only - Perfect fit
    mask = None
    if automasker:
        try:
            mask = automasker(person, cloth_type="overall")[0].resize((768, 1024))
        except Exception as e:
            print(f"Automasker error: {e}")
            mask = None

    if mask is None:
        mask = Image.new("L", (768, 1024), 0)
        ImageDraw.Draw(mask).rectangle([76, 300, 692, 1024], fill=255)

    # Generate
    try:
        if torch is None or pipeline is None:
            raise RuntimeError("PyTorch or CatVTON pipeline is not loaded.")
        with torch.inference_mode():
            gen = pipeline(
                image=person,
                condition_image=cloth,
                mask=mask,
                num_inference_steps=40,
                guidance_scale=2.5,
                seed=42
            )[0]
    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"status": "error", "message": f"Generation failed: {str(e)}"}), 500

    # FACE/BG/HAIR SAME - Key!
    final = Image.composite(gen, person, mask)
    face_h = int(1024 * 0.32)
    final.paste(person_orig.crop((0, 0, 768, face_h)), (0, 0))

    result_data = encode(final)
    return jsonify({
        "status": "success",
        "result_image": result_data,
        "resultImage": result_data
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)