# NVIDIA'nın PyTorch için optimize ettiği Docker imajı
FROM nvcr.io/nvidia/pytorch:23.05-py3

# Çalışma dizinini ayarla
WORKDIR /app

# Gerekli bağımlılıkların kurulumu için requirements.txt oluştur
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . /app/

# Gradio uygulamasını başlat
CMD ["python", "app.py"]
