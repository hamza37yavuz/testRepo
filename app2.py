import gradio as gr
from diffusers import FluxPipeline
from deep_translator import GoogleTranslator
import torch

# FLUX Pipeline yükleme

def load_flux_pipeline():
    try:
        print("FLUX Pipeline yükleniyor...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        print("Model başarıyla yüklendi.")
        return pipe
    except Exception as e:
        print(f"Model yükleme sırasında hata oluştu: {e}")
        return None

pipe = load_flux_pipeline()

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
    if pipe is None:
        return "Model yüklenemedi, lütfen tekrar deneyin.", None

    if len(prompt) > 400:
        return "Prompt çok uzun! Lütfen 400 karakterden kısa bir şey girin.", None
    if width > 512 or height > 512:
        return "Boyutlar sınırı aşıyor! Maksimum boyut 512x512 olmalıdır.", None

    try:
        # Türkçe promptu İngilizce'ye çevir
        prompt = translate_to_english(prompt)

        # Görseli üret
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=3.5,
            height=height,
            width=width,
            num_inference_steps=50
        ).images[0]
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
with gr.Blocks() as demo:
    gr.Markdown("### FLUX Görsel Üretimi")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt (Türkçe)", placeholder="Bir şey yazın (max 400 karakter)")
        width = gr.Slider(label="Genişlik", minimum=128, maximum=512, step=64, value=512)
        height = gr.Slider(label="Yükseklik", minimum=128, maximum=512, step=64, value=512)

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

demo.launch(server_name="0.0.0.0", server_port=7860)
