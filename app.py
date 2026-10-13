# pyright: reportMissingImports=false
# pyright: reportUnusedImport=false
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os, io, uuid, tempfile, base64
from PIL import Image

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None
    handle_file = None

app = FastAPI(title="VASTRA FREE UNLIMITED")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def decode_base64_to_file(b64_str: str, output_path: str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    with open(output_path, "wb") as f:
        f.write(img_data)

def encode_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

@app.get("/")
def home():
    return {
        "engine": "VASTRA FREE UNLIMITED - Face Same + Dress Fit",
        "status": "ok",
        "gpu": "HF ZeroGPU - 100% Free Cloud Inference"
    }

@app.post("/tryon")
async def tryon(
    request: Request,
    person: UploadFile = File(None),
    cloth: UploadFile = File(None)
):
    tmp_dir = tempfile.gettempdir()
    pid = str(uuid.uuid4())[:8]
    person_path = os.path.join(tmp_dir, f"{pid}_person.jpg")
    cloth_path = os.path.join(tmp_dir, f"{pid}_cloth.jpg")

    try:
        # 1. Extract images from either Multipart Form-Data OR JSON payload
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
            p_b64 = data.get("person_image") or data.get("personImage")
            c_b64 = data.get("cloth_image") or data.get("garmentImage")
            if not p_b64 or not c_b64:
                return {"status": "error", "message": "person and cloth images required"}
            decode_base64_to_file(p_b64, person_path)
            decode_base64_to_file(c_b64, cloth_path)
        else:
            if not person or not cloth:
                return {"status": "error", "message": "person and cloth files required"}
            with open(person_path, "wb") as f:
                f.write(await person.read())
            with open(cloth_path, "wb") as f:
                f.write(await cloth.read())

        print("Connecting to Hugging Face Free Space...")
        if Client is None or handle_file is None:
            return {"status": "error", "message": "gradio_client not installed on server"}

        generated_file = None

        # TRY 1: Leffa - Best quality & stability
        try:
            client = Client("franciszzj/Leffa")
            result = client.predict(
                src_image_path=handle_file(person_path),
                ref_image_path=handle_file(cloth_path),
                ref_accelerate=False,
                step=30,
                scale=2.5,
                seed=42,
                vt_model_type="viton_hd",
                vt_garment_type="upper_body",
                vt_reward_type="t2i",
                api_name="/leffa_predict_vt"
            )
            print(f"Leffa Success: {result}")
            if isinstance(result, (list, tuple)):
                generated_file = result[0]
            else:
                generated_file = result
        except Exception as e1:
            print(f"Leffa failed: {e1}, attempting fallback to IDM-VTON...")

            # TRY 2: IDM-VTON Fallback
            client2 = Client("yisol/IDM-VTON")
            result2 = client2.predict(
                dict={"background": handle_file(person_path), "layers": [], "composite": None},
                garm_img=handle_file(cloth_path),
                garment_des="saree, upper body garment",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            print(f"IDM-VTON Success: {result2}")
            if isinstance(result2, (list, tuple)):
                generated_file = result2[0]
            else:
                generated_file = result2

        if not generated_file or not os.path.exists(generated_file):
            return {
                "status": "error",
                "message": "AI Space did not return an output file."
            }

        result_b64 = encode_file_to_base64(generated_file)
        return {
            "status": "success",
            "result_image": result_b64,
            "resultImage": result_b64,
            "image": result_b64
        }

    except Exception as e:
        print(f"Try-On Error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "tip": "HF Space is currently busy. Please retry in a few seconds."
        }
    finally:
        # Cleanup temp uploaded files
        for p in (person_path, cloth_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)