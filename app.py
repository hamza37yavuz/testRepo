import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusion3Pipeline
from deep_translator import GoogleTranslator
import torch

class StableDiffusionApp:
    def __init__(self):
        self.pipe = self.load_pipeline()
        self.negative_prompt = (
            "violence, explicit content, gore, inappropriate for children, "
            "blood, weapon, nudity"
        )

    def load_pipeline(self):
        try:
            print("Stable Diffusion 3.5 Large modeli yükleniyor...")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                torch_dtype=torch.bfloat16
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)
            print(f"Model {device.upper()} üzerinde çalıştırılıyor.")
            return pipe
        except Exception as e:
            print(f"Model yükleme sırasında hata oluştu: {e}")
            return None

    def translate_to_english(self, prompt):
        try:
            translated_text = GoogleTranslator(source="turkish", target="english").translate(prompt)
            return translated_text
        except Exception as e:
            print(f"Çeviri sırasında hata: {e}")
            return prompt

    def generate_image(self, prompt, width, height, translate):
        if self.pipe is None:
            return "Model yüklenemedi, lütfen tekrar deneyin.", None

        if len(prompt) > 200:
            return "Prompt çok uzun! Lütfen 200 karakterden kısa bir şey girin.", None
        if width > 400 or height > 400:
            return "Boyutlar sınırı aşıyor! Maksimum boyut 400x400 olmalıdır.", None

        try:
            if translate:
                prompt = self.translate_to_english(prompt)

            image = self.pipe(prompt, width=width, height=height, negative_prompt=self.negative_prompt).images[0]
            return None, image
        except Exception as e:
            print(f"Görsel üretim sırasında hata oluştu: {e}")
            return "Görsel üretim sırasında bir hata oluştu.", None

    def check_device(self):
        if torch.cuda.is_available():
            return "GPU kullanılıyor."
        else:
            return "GPU kullanılmıyor, CPU üzerinde çalışıyor."

    def launch_app(self):
        with gr.Blocks(css="body { background-color: #f0f8ff; font-family: Arial, sans-serif; } .gr-button { background-color: #ff7f50; color: white; border: none; }") as demo:
            gr.Markdown("### Stable Diffusion Görsel Üretim Aracı")

            with gr.Row():
                prompt = gr.Textbox(label="Prompt (Türkçe)", placeholder="Bir şey yazın (max 200 karakter)")
                width = gr.Slider(label="Genişlik", minimum=100, maximum=400, step=50, value=400)
                height = gr.Slider(label="Yükseklik", minimum=100, maximum=400, step=50, value=400)

            with gr.Row():
                translate = gr.Checkbox(label="Türkçe'den İngilizce'ye çevir", value=True)

            with gr.Row():
                device_status = gr.Textbox(label="Cihaz Durumu", value=self.check_device(), interactive=False)

            output_text = gr.Textbox(label="Hata Mesajı")
            output_image = gr.Image(label="Üretilen Görsel")

            generate_button = gr.Button("Görsel Üret")
            generate_button.click(
                fn=self.generate_image,
                inputs=[prompt, width, height, translate],
                outputs=[output_text, output_image]
            )

        demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    app = StableDiffusionApp()
    app.launch_app()
