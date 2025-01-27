import gradio as gr
from diffusers import DiffusionPipeline
import torch
from huggingface_hub import login
import sys

class FluxImageGenerator:
    def __init__(self, model_id="black-forest-labs/FLUX.1-dev", device="cuda"):
        """
        FLUX.1-dev modelini yükler ve ayarlar.
        
        Args:
            model_id (str): Hugging Face model kimliği.
            device (str): Kullanılacak cihaz ("cuda" veya "cpu").
        """
        self.model_id = model_id
        self.device = device
        self.pipe = None  # Model başlangıçta yüklenmemiş olacak

    def _load_model(self, token):
        """
        Modeli yükler ve belirtilen cihaza taşır.
        
        Args:
            token (str): Hugging Face token'ı.
        """
        print(f"Model yükleniyor: {self.model_id}")
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_auth_token=token  # Token'ı kullanarak modeli yükle
        )
        self.pipe = self.pipe.to(self.device)
        print("Model başarıyla yüklendi!")

    def generate_image(self, prompt, negative_prompt=""):
        """
        Metin girdisine dayalı olarak görsel oluşturur.
        
        Args:
            prompt (str): İstenen görseli tanımlayan metin.
            negative_prompt (str): İstenmeyen öğeleri tanımlayan metin.
        
        Returns:
            PIL.Image: Oluşturulan görsel.
        """
        if self.pipe is None:
            raise gr.Error("Lütfen önce modeli yükleyin!")
        
        print(f"Görsel oluşturuluyor: {prompt}")
        with torch.autocast(self.device):
            image = self.pipe(prompt, negative_prompt=negative_prompt).images[0]
        print("Görsel başarıyla oluşturuldu!")
        return image

def create_gradio_interface(generator):
    """
    Gradio arayüzünü oluşturur.
    
    Returns:
        gr.Blocks: Gradio arayüzü.
    """

    def generate_image_wrapper(prompt, negative_prompt):
        # Prompt'un 300 karakteri aşmasını engelle
        if len(prompt) > 300:
            raise gr.Error("Prompt 300 karakteri geçemez! Lütfen daha kısa bir metin girin.")
        return generator.generate_image(prompt, negative_prompt)

    # Renkli ve çocuk dostu tema
    custom_theme = gr.themes.Default(
        primary_hue="teal",  # Ana renk
        secondary_hue="pink",  # İkincil renk
        neutral_hue="gray",  # Nötr renk
        font=gr.themes.GoogleFont("Comic Neue"),  # Eğlenceli bir yazı tipi
    )

    with gr.Blocks(theme=custom_theme) as demo:
        gr.Markdown("# 🎨 FLUX.1-dev ile Eğlenceli Görsel Oluşturma 🎨")
        gr.Markdown("### Çocuklar için renkli ve eğlenceli görseller oluşturun!")
        
        # Görsel oluşturma alanı
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt (İstenen Görsel)",
                    placeholder="Örneğin: Mutlu bir çocuk, renkli balonlarla",
                    max_lines=3,  # Metin kutusunun boyutunu sınırla
                    max_length=300,  # Maksimum 300 karakter
                    info="Lütfen 300 karakteri geçmeyen bir metin girin.",  # Bilgilendirme mesajı
                )
                negative_prompt = gr.Textbox(
                    label="Negatif Prompt (İstenmeyen Öğeler)",
                    placeholder="Örneğin: şiddet, korku, çıplaklık",
                    max_lines=2,  # Metin kutusunun boyutunu sınırla
                )
                generate_button = gr.Button("Görsel Oluştur", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Oluşturulan Görsel", interactive=False)

        # Buton işlevleri
        generate_button.click(
            generate_image_wrapper,
            inputs=[prompt, negative_prompt],
            outputs=output_image,
        )

        gr.Markdown("### Örnek Promptlar:")
        gr.Examples(
            examples=[
                ["Mutlu bir çocuk, yeşil bir ormanda, güneş ışığı altında, renkli kelebeklerle çevrili", "şiddet, korku"],
                ["Renkli balonlarla dolu bir parti, mutlu çocuklar, pastel renkler", "karanlık, üzüntü"],
                ["Bir uzay gemisi, yıldızlar, gezegenler, renkli ışıklar", "şiddet, korku"],
            ],
            inputs=[prompt, negative_prompt],
            label="Örnekler"
        )

    return demo

def main():
    # Komut satırından token alın
    if len(sys.argv) != 2:
        print("Kullanım: python app2.py <HuggingFace_Token>")
        sys.exit(1)

    HUGGINGFACE_TOKEN = sys.argv[1]

    # Hugging Face hesabına giriş yap
    login(token=HUGGINGFACE_TOKEN)

    # Modeli yükleyin
    generator = FluxImageGenerator(device="cuda")
    generator._load_model(HUGGINGFACE_TOKEN)

    # Gradio arayüzünü oluştur ve başlat
    demo = create_gradio_interface(generator)
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
