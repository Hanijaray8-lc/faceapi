FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/Zheng-Chong/CatVTON.git .
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install flask pillow diffusers transformers accelerate
COPY app.py ./app.py
CMD ["python", "app.py"]