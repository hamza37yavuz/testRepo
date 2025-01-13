import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
from deep_translator import GoogleTranslator
import torch

# Stable Diffusion Pipeline'i yükleme
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to(device)  # GPU veya CPU üzerinde çalıştırma

# Çeviri fonksiyonu
def translate_to_english(prompt):
    try:
        # Türkçe prompt'u İngilizce'ye çevir
        translated_text = GoogleTranslator(source="turkish", target="english").translate(prompt)
        return translated_text
    except Exception as e:
        print(f"Çeviri sırasında hata: {e}")
        return prompt

# Görsel üretim fonksiyonu
negative_prompt = (
    "violence, explicit content, gore, inappropriate for children, "
    "blood, weapon, nudity"
)

def generate_image(prompt, width, height):
    if len(prompt) > 200:
        return "Prompt çok uzun! Lütfen 200 karakterden kısa bir şey girin.", None
    if width > 400 or height > 400:
        return "Boyutlar sınırı aşıyor! Maksimum boyut 400x400 olmalıdır.", None

    # Türkçe promptu İngilizce'ye çevir
    prompt = translate_to_english(prompt)

    # Görseli üret
    image = pipe(prompt, width=width, height=height, negative_prompt=negative_prompt).images[0]
    return None, image

# GPU veya CPU durumunu kontrol eden fonksiyon
def check_device():
    if torch.cuda.is_available():
        return "GPU kullanılıyor."
    else:
        return "GPU kullanılmıyor, CPU üzerinde çalışıyor."

# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("### Stable Diffusion Çocuklara Özel Görsel Üretimi")
    
    with gr.Row():
        prompt = gr.Textbox(label="Prompt (Türkçe)", placeholder="Bir şey yazın (max 200 karakter)")
        width = gr.Slider(label="Genişlik", minimum=100, maximum=400, step=50, value=400)
        height = gr.Slider(label="Yükseklik", minimum=100, maximum=400, step=50, value=400)

    with gr.Row():
        device_status = gr.Textbox(label="Cihaz Durumu", value=check_device(), interactive=False)

    output_text = gr.Textbox(label="Hata Mesajı")
    output_image = gr.Image(label="Üretilen Görsel")

    generate_button = gr.Button("Görsel Üret")
    generate_button.click(
        fn=generate_image,
        inputs=[prompt, width, height],
        outputs=[output_text, output_image]
    )

demo.launch()