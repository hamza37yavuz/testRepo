import gradio as gr
from diffusers import DiffusionPipeline
import torch
from huggingface_hub import login
import sys
from googletrans import Translator  # Prompt çevirisi için

class StableDif: 
    def __init__(self, model_id="stabilityai/stable-diffusion-3-large", devices=None, image_size=128):
        """
        Stable Diffusion 3.5 Large modelini yükler ve ayarlar.
        
        Args:
            model_id (str): Hugging Face model kimliği.
            devices (list): Kullanılacak GPU'ların listesi (ör. ["cuda:0", "cuda:1"]).
            image_size (int): Görsel boyutu (128, 256 veya 512).
        """
        self.model_id = model_id
        self.devices = devices or ["cuda:0"]  # Varsayılan olarak tek GPU
        self.image_size = image_size if image_size in [128, 256, 512] else 128  # Varsayılan 128
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
            
            self.pipe = torch.nn.DataParallel(self.pipe, device_ids=[torch.device(device) for device in self.devices])
            print("[DEBUG] Model başarıyla yüklendi!")
        except Exception as e:
            print(f"[ERROR] Model yükleme başarısız oldu: {e}")
            raise RuntimeError(f"Model yüklenemedi: {e}")

    def generate_image(self, prompt, translate_prompt=False):
        """
        Metin girdisine dayalı olarak görsel oluşturur.
        
        Args:
            prompt (str): İstenen görseli tanımlayan metin.
            translate_prompt (bool): Prompt'u İngilizceye çevir.
        
        Returns:
            PIL.Image: Oluşturulan görsel.
        """
        negative_prompt = (
            "violence, gore, nudity, explicit content, blood, horror, weapons, "
            "disturbing imagery, adult content, offensive material"
        )

        if self.pipe is None:
            raise gr.Error("[ERROR] Model yüklenmeden görsel oluşturulamaz!")

        try:
            if translate_prompt:
                translator = Translator()
                prompt = translator.translate(prompt, dest="en").text
                print(f"[DEBUG] Çevrilen Prompt: {prompt}")

            print(f"[DEBUG] Görsel oluşturuluyor: {prompt}")
            with torch.autocast(self.devices[0]):
                image = self.pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    height=self.image_size,
                    width=self.image_size
                ).images[0]
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
    def generate_image_wrapper(prompt, translate_prompt):
        return generator.generate_image(prompt, translate_prompt)

    custom_css = """
    .gradio-container {
        background: linear-gradient(45deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1, #fad0c4, #ffdde1);
    }
    """

    try:
        with gr.Blocks(css=custom_css) as demo:
            gr.Markdown("# 🎨 Stable Diffusion 3.5 Large ile Görsel Oluşturma 🎨")
            gr.Markdown("### Renkli ve eğlenceli görseller oluşturun!")
            
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Prompt (İstenen Görsel)",
                        placeholder="Örneğin: Mutlu bir çocuk, renkli balonlarla",
                        max_lines=3,
                        info="Lütfen istediğiniz görseli tanımlayan bir metin girin."
                    )
                    translate_prompt = gr.Checkbox(
                        label="Prompt'u İngilizceye Çevir",
                        value=False,
                        info="İşaretli değilse, prompt otomatik olarak İngilizceye çevrilir."
                    )
                    generate_button = gr.Button("Görsel Oluştur", variant="primary")
                with gr.Column():
                    output_image = gr.Image(label="Oluşturulan Görsel", interactive=False)

            generate_button.click(
                generate_image_wrapper,
                inputs=[prompt, translate_prompt],
                outputs=output_image,
            )
        return demo
    except Exception as e:
        print(f"[ERROR] Gradio arayüzü oluşturulamadı: {e}")
        raise RuntimeError(f"Gradio arayüzü başlatılamadı: {e}")
      
def main():
    if len(sys.argv) < 2:
        print("Kullanım: python app2.py <HuggingFace_Token> [image_size]")
        sys.exit(1)

    HUGGINGFACE_TOKEN = sys.argv[1]
    IMAGE_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] in ["128", "256", "512"] else 128

    try:
        print("[DEBUG] Hugging Face hesabına giriş yapılıyor...")
        login(token=HUGGINGFACE_TOKEN)
        print("[DEBUG] Hugging Face giriş başarılı!")

        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not devices:
            raise RuntimeError("Kullanılabilir GPU bulunamadı.")
        generator = StableDif(devices=devices, image_size=IMAGE_SIZE)
        generator._load_model(HUGGINGFACE_TOKEN)

        demo = create_gradio_interface(generator)
        print("[DEBUG] Gradio arayüzü başlatılıyor...")
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"[ERROR] Program başlatılamadı: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
