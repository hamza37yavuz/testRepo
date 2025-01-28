import gradio as gr
from diffusers import DiffusionPipeline
import torch
from huggingface_hub import login
import sys

class FluxImageGenerator:
    def __init__(self, model_id="black-forest-labs/FLUX.1-dev", devices=None):
        """
        FLUX.1-dev modelini yükler ve ayarlar.
        
        Args:
            model_id (str): Hugging Face model kimliği.
            devices (list): Kullanılacak GPU'ların listesi (ör. ["cuda:0", "cuda:1"]).
        """
        self.model_id = model_id
        self.devices = devices or ["cuda:0"]  # Varsayılan olarak tek GPU
        self.pipe = None  # Model başlangıçta yüklenmemiş olacak

    def _load_model(self, token):
        """
        Modeli yükler ve belirtilen GPU'lara dağıtır.
        
        Args:
            token (str): Hugging Face token'ı.
        """
        try:
            print(f"[DEBUG] Model yükleniyor: {self.model_id}")
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                use_auth_token=token  # Token'ı kullanarak modeli yükle
            )
            
            # Çoklu GPU kullanımı (DataParallel)
            if len(self.devices) > 1:
                print(f"[DEBUG] {len(self.devices)} GPU kullanılıyor: {', '.join(self.devices)}")
                self.pipe.enable_model_cpu_offload()  # Bellek optimizasyonu için
            else:
                print(f"[DEBUG] Tek GPU kullanılıyor: {self.devices[0]}")
                self.pipe = self.pipe.to(self.devices[0])

            print("[DEBUG] Model başarıyla yüklendi!")
        except Exception as e:
            print(f"[ERROR] Model yükleme başarısız oldu: {e}")
            raise RuntimeError(f"Model yüklenemedi: {e}")

    def generate_image(self, prompt):
        """
        Metin girdisine dayalı olarak görsel oluşturur.
        
        Args:
            prompt (str): İstenen görseli tanımlayan metin.
        
        Returns:
            PIL.Image: Oluşturulan görsel.
        """
        negative_prompt = "şiddet, korku, çıplaklık"  # Sabit negatif prompt
        if self.pipe is None:
            raise gr.Error("[ERROR] Model yüklenmeden görsel oluşturulamaz!")

        try:
            print(f"[DEBUG] Görsel oluşturuluyor: {prompt}")
            with torch.autocast(self.devices[0]):
                image = self.pipe(prompt, negative_prompt=negative_prompt, height=512, width=512).images[0]
            print("[DEBUG] Görsel başarıyla oluşturuldu!")
            return image
        except Exception as e:
            print(f"[ERROR] Görsel oluşturulurken hata oluştu: {e}")
            raise RuntimeError(f"Görsel oluşturulamadı: {e}")

def create_gradio_interface(generator):
    """
    Gradio arayüzünü oluşturur.
    
    Returns:
        gr.Blocks: Gradio arayüzü.
    """

    def generate_image_wrapper(prompt):
        # Prompt'un 300 karakteri aşmasını engelle
        if len(prompt) > 300:
            raise gr.Error("Prompt 300 karakteri geçemez! Lütfen daha kısa bir metin girin.")
        return generator.generate_image(prompt)

    # Renkli ve çocuk dostu tema
    custom_theme = gr.themes.Default(
        primary_hue="teal",  # Ana renk
        secondary_hue="pink",  # İkincil renk
        neutral_hue="gray",  # Nötr renk
        font=gr.themes.GoogleFont("Comic Neue"),  # Eğlenceli bir yazı tipi
    )

    try:
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
                    generate_button = gr.Button("Görsel Oluştur", variant="primary")
                with gr.Column():
                    output_image = gr.Image(label="Oluşturulan Görsel", interactive=False)

            # Buton işlevleri
            generate_button.click(
                generate_image_wrapper,
                inputs=[prompt],
                outputs=output_image,
            )

            gr.Markdown("### Örnek Promptlar:")
            gr.Examples(
                examples=[
                    ["Mutlu bir çocuk, yeşil bir ormanda, güneş ışığı altında, renkli kelebeklerle çevrili"],
                    ["Renkli balonlarla dolu bir parti, mutlu çocuklar, pastel renkler"],
                    ["Bir uzay gemisi, yıldızlar, gezegenler, renkli ışıklar"],
                ],
                inputs=[prompt],
                label="Örnekler"
            )
        return demo
    except Exception as e:
        print(f"[ERROR] Gradio arayüzü oluşturulamadı: {e}")
        raise RuntimeError(f"Gradio arayüzü başlatılamadı: {e}")

def main():
    # Komut satırından token alın
    if len(sys.argv) != 2:
        print("Kullanım: python app2.py <HuggingFace_Token>")
        sys.exit(1)

    HUGGINGFACE_TOKEN = sys.argv[1]

    try:
        # Hugging Face hesabına giriş yap
        print("[DEBUG] Hugging Face hesabına giriş yapılıyor...")
        login(token=HUGGINGFACE_TOKEN)
        print("[DEBUG] Hugging Face giriş başarılı!")

        # Modeli yükleyin (birden fazla GPU kullanımı için ayar)
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not devices:
            raise RuntimeError("Kullanılabilir GPU bulunamadı.")
        generator = FluxImageGenerator(devices=devices)
        generator._load_model(HUGGINGFACE_TOKEN)

        # Gradio arayüzünü oluştur ve başlat
        demo = create_gradio_interface(generator)
        print("[DEBUG] Gradio arayüzü başlatılıyor...")
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"[ERROR] Program başlatılamadı: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
