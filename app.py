import gradio as gr
from google.cloud import translate_v2 as translate
from diffusers import StableDiffusionPipeline
import torch

# Google Translate API istemcisi
translate_client = translate.Client()

# Stable Diffusion Pipeline'i yükleme
def load_pipeline():
    try:
        print("Stable Diffusion Pipeline yükleniyor...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            cache_dir="./cache",
            resume_download=True,
            timeout=120
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"Model {device.upper()} üzerinde çalıştırılıyor.")
        return pipe
    except Exception as e:
        print(f"Model yükleme sırasında hata oluştu: {e}")
        return None

pipe = load_pipeline()

# Çeviri fonksiyonu (Google Translate API)
def translate_to_english_google_api(prompt):
    try:
        result = translate_client.translate(prompt, source_language="tr", target_language="en")
        return result["translatedText"]
    except Exception as e:
        print(f"Çeviri sırasında hata oluştu: {e}")
        return prompt

# Görsel üretim fonksiyonu
negative_prompt = (
    "violence, explicit content, gore, inappropriate for children, "
    "blood, weapon, nudity"
)

def generate_image(prompt, width, height, translate_prompt):
    if pipe is None:
        return "Model yüklenemedi, lütfen tekrar deneyin.", None

    if len(prompt) > 200:
        return "Prompt çok uzun! Lütfen 200 karakterden kısa bir şey girin.", None
    if width > 400 or height > 400:
        return "Boyutlar sınırı aşıyor! Maksimum boyut 400x400 olmalıdır.", None

    try:
        # Prompt çevirme seçeneği
        if translate_prompt:
            prompt = translate_to_english_google_api(prompt)

        # Görseli üret
        image = pipe(prompt, width=width, height=height, negative_prompt=negative_prompt).images[0]
        return None, image
    except Exception as e:
        print(f"Görsel üretim sırasında hata oluştu: {e}")
        return "Görsel üretim sırasında bir hata oluştu.", None

# GPU veya CPU durumunu kontrol eden fonksiyon
def check_device():
    if torch.cuda.is_available():
        return "GPU kullanılıyor."
    else:
        return "GPU kullanılmıyor, CPU üzerinde çalışıyor."

# Gradio arayüzü
with gr.Blocks(css="body {background: linear-gradient(135deg, #ff7eb3, #ff758c, #ffd5cd);}") as demo:
    gr.Markdown("### 🎨 Stable Diffusion: Yapay Zeka Eğitim Görsel Üretimi")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt (Türkçe)", placeholder="Bir şey yazın (max 200 karakter)")
        width = gr.Slider(label="Genişlik", minimum=100, maximum=400, step=50, value=400)
        height = gr.Slider(label="Yükseklik", minimum=100, maximum=400, step=50, value=400)

    with gr.Row():
        translate_prompt = gr.Checkbox(label="İngilizce'ye çevir (Google Translate API)", value=True)
        device_status = gr.Textbox(label="Cihaz Durumu", value=check_device(), interactive=False)

    output_text = gr.Textbox(label="Hata Mesajı")
    output_image = gr.Image(label="Üretilen Görsel")

    generate_button = gr.Button("Görsel Üret")
    generate_button.click(
        fn=generate_image,
        inputs=[prompt, width, height, translate_prompt],
        outputs=[output_text, output_image]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
