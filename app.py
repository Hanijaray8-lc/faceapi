from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
import os, uuid, shutil, base64

app = FastAPI(title="VASTRA FREE UNLIMITED")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "engine": "VASTRA FREE UNLIMITED - Face Same + Dress Fit",
        "status": "ok",
        "gpu": "HF ZeroGPU - No Render GPU needed"
    }

@app.post("/tryon")
async def tryon(person: UploadFile = File(...), cloth: UploadFile = File(...)):
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    pid = str(uuid.uuid4())[:8]

    person_path = f"{tmp_dir}/{pid}_person.jpg"
    cloth_path = f"{tmp_dir}/{pid}_cloth.jpg"

    try:
        # Save files
        with open(person_path, "wb") as f:
            f.write(await person.read())
        with open(cloth_path, "wb") as f:
            f.write(await cloth.read())

        print("Connecting to HF Free Space...")

        # TRY 1: Leffa - Most stable & fast for saree/dress
        try:
            client = Client("franciszzj/Leffa", download_files=False)
            result = client.predict(
                src_image_path=person_path,
                ref_image_path=cloth_path,
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
            # result is image path url
            return {"image": result, "status": "success", "engine": "Leffa HD - Free"}

        except Exception as e1:
            print(f"Leffa failed: {e1}, trying IDM-VTON...")

            # TRY 2: IDM-VTON Fallback
            client2 = Client("yisol/IDM-VTON")
            result2 = client2.predict(
                dict={"background": open(person_path, "rb"), "layers": [], "composite": None},
                garm_img=open(cloth_path, "rb"),
                garment_des="saree, upper body garment",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            print(f"IDM-VTON Success: {result2}")
            return {"image": result2[0] if isinstance(result2, list) else result2, "status": "success", "engine": "IDM-VTON - Free"}

    except Exception as e:
        print(f"FINAL ERROR: {e}")
        return {
            "error": str(e),
            "status": "retry",
            "tip": "HF Space busy, queue la irukku ma - 20 sec kalichu thirumba try pannu ma"
        }
    finally:
        # Cleanup
        try:
            os.remove(person_path)
            os.remove(cloth_path)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)