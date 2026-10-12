from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import torch, io, base64, os

device = "cpu"
app = Flask(__name__)

print("Loading VASTRA...")
try:
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    automasker = AutoMasker(device=device)
    pipeline = CatVTONPipeline(base_ckpt="runwayml/stable-diffusion-inpainting", attn_ckpt="zhengchong/CatVTON", attn_ckpt_version="mix", weight_dtype=torch.float32, device=device)
except Exception as e:
    print(f"Model load: {e}")
    automasker = None
    pipeline = None

def decode(b64):
    if ',' in b64: b64=b64.split(',')[1]
    img=Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return img.resize((512,682), Image.LANCZOS)

def encode(img):
    buf=io.BytesIO(); img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()

@app.route("/")
def home(): return jsonify({"status":"ok","engine":"VASTRA FREE UNLIMITED - Face Same + Dress Fit"})

@app.route("/tryon", methods=["POST"])
def tryon():
    data=request.get_json(force=True)
    p_b64=data.get("person_image") or data.get("personImage")
    c_b64=data.get("cloth_image") or data.get("garmentImage")

    person_orig=decode(p_b64)
    person=person_orig.copy()
    cloth=decode(c_b64)

    # Dress area mattum mask
    if automasker:
        try: mask=automasker(person, cloth_type="overall")[0].resize((512,682))
        except:
            mask=Image.new("L",(512,682),0); ImageDraw.Draw(mask).rectangle([51,250,461,682],fill=255)
    else:
        mask=Image.new("L",(512,682),0); ImageDraw.Draw(mask).rectangle([51,250,461,682],fill=255)

    # Generate
    with torch.inference_mode():
        gen=pipeline(image=person, condition_image=cloth, mask=mask, num_inference_steps=15, guidance_scale=2.0, seed=42)[0]

    # FACE/BG/HAIR SAME - Key!
    final=Image.composite(gen, person, mask)
    face_h=int(682*0.32)
    final.paste(person_orig.crop((0,0,512,face_h)),(0,0))

    return jsonify({"status":"success","result_image":encode(final),"resultImage":encode(final)})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",10000)))