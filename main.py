import os
import subprocess
from pathlib import Path
import shutil

import torch
from pytorch_lightning import LightningModule
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import time

print(torch.__version__)

class PredictionRequest(BaseModel):
    input_data: list

class Item(BaseModel):
    text: str
    reference_audio: str
    reference_text: str
    streaming: bool = False

class MyModel(LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model layers and components here
        # e.g., self.layer = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        # Define the forward pass
        # e.g., return self.layer(x)
        return x

def load_model(model_path):
    model = MyModel()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Load checkpoint
    model.load_state_dict(checkpoint, strict=False)  # Load model weights
    model.eval()  # Set model to evaluation mode
    return model

model_path = "checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"  # Adjust path to the exact model checkpoint file

model = load_model(model_path)

def predict(model, input_data):
    with torch.no_grad():
        predictions = model(input_data)
    return predictions

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Fish Speech API!"}

@app.on_event("startup")
async def startup_event():
    # 启动fish-speech的WebUI
    command = [
        "python", "-m", "tools.webui",
        "--llama-checkpoint-path", "checkpoints/fish-speech-1.2-sft",
        "--decoder-checkpoint-path", "checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
        "--decoder-config-name", "firefly_gan_vq"
    ]
    subprocess.Popen(command)
    time.sleep(20)  # 等待外部服务启动，必要时调整时间

@app.post("/predict/")
async def make_prediction(request: PredictionRequest):
    input_tensor = torch.Tensor(request.input_data)  # Convert to tensor
    prediction = predict(model, input_tensor)
    return {"prediction": prediction.tolist()}

@app.post("/synthesize/")
async def synthesize_audio(reference_audio: UploadFile = File(...), text: str = Form(...)):
    temp_audio_path = Path("temp") / reference_audio.filename
    temp_audio_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(reference_audio.file, buffer)

    # 确认外部服务可用性
    print("Checking target service availability...")
    try:
        response = requests.post("http://127.0.0.1:8000/synthesize", json={
            "data": ["test"]
        })
        if response.status_code != 200:
            print(f"Target service returned non-200 status code: {response.status_code}")
            raise Exception("Target service not available")
    except requests.ConnectionError:
        print("Failed to connect to target service")
        raise HTTPException(status_code=500, detail="Target service not reachable")

    try:
        output_audio_path = "output/output.wav"
        command = [
            "python", "tools/post_api.py",
            "--text", text,
            "--reference_audio", str(temp_audio_path),
            "--reference_text", text,
            "--streaming", "True"
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Model execution failed: " + result.stderr)

        def iterfile():
            with open(output_audio_path, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(iterfile(), media_type="audio/wav")

    except Exception as e:
        print(f"Exception in /synthesize/: {str(e)}")  # 打印异常信息到控制台
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        temp_audio_path.unlink(missing_ok=True)
        Path(output_audio_path).unlink(missing_ok=True)






def test_api():
    # 验证/predict/端点
    try:
        url = "http://127.0.0.1:8000/predict/"
        payload = {"input_data": [1.0, 2.0, 3.0]}
        response = requests.post(url, json=payload)
        print("Prediction response:", response.json())
    except Exception as e:
        print(f"Prediction request failed: {str(e)}")

    # 验证/synthesize/端点
    try:
        url = "http://127.0.0.1:8000/synthesize/"
        files = {'reference_audio': open('audio/myaud.wav', 'rb')}
        data = {'text': '力合创投集团有限公司是深圳清华大学研究院精心打造和培育的科技创新服务平台，成立于1999年8月，国家高新技术企业。'}
        response = requests.post(url, files=files, data=data)
        print("Synthesize response status code:", response.status_code)
        if response.status_code == 200:
            print("Synthesize response:", response.json())
        else:
            print("Synthesize failed with response:", response.text)
    except Exception as e:
        print(f"Synthesize request failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import threading

    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    # 等待服务器启动完成
    def wait_for_server():
        while True:
            try:
                response = requests.get("http://127.0.0.1:8000")
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(1)  # 每隔1秒检查一次

    wait_for_server()

    # 服务器启动后测试API
    test_api()
