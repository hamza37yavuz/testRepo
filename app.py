import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
from deep_translator import GoogleTranslator
import torch

# Stable Diffusion Pipeline'i yükleme
def load_pipeline():
    try:
        print("Stable Diffusion Pipeline yükleniyor...")
        # Modeli önbelleğe kaydet ve zaman aşımı ile yükle
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            cache_dir="./cache",  # Modelleri cache'e kaydet
            resume_download=True,  # İndirme kesilirse devam et
            timeout=120  # Zaman aşımını artır
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"Model {device.upper()} üzerinde çalıştırılıyor.")
        return pipe
    except Exception as e:
        print(f"Model yükleme sırasında hata oluştu: {e}")
        return None

pipe = load_pipeline()

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
    
    if len(prompt) > 200:
        return "Prompt çok uzun! Lütfen 200 karakterden kısa bir şey girin.", None
    if width > 400 or height > 400:
        return "Boyutlar sınırı aşıyor! Maksimum boyut 400x400 olmalıdır.", None

    try:
        # Türkçe promptu İngilizce'ye çevir
        prompt = translate_to_english(prompt)

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
with gr.Blocks(css="body {background-color: #f0f8ff;} .footer {text-align: center; margin-top: 20px;}") as demo:
    gr.Markdown("""<h1 style='text-align: center;'>Stable Diffusion Çocuklara Özel Görsel Üretimi</h1>""")

    # Logo Ekleme
    with gr.Row():
        gr.Image("logo.ico", elem_id="logo", interactive=False, label="")

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

    # Footer için bir Markdown ekleme
    with gr.Row(elem_id="footer"):
        gr.Markdown("<p>Bu uygulama <b>Gradio</b> kullanılarak oluşturulmuştur.</p>")

# Uygulama başlatma
demo.launch(server_name="0.0.0.0", server_port=7860)
