import gradio as gr
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

# ✅ 设置设备（使用 GPU 加速）
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 加载 BLIP-2 视觉模型
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float32).to(device)

# ✅ 加载 MedLLaMA 2 语言模型
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float32, device_map="auto")

# ✅ 解析医学影像
def generate_image_description(image):
    image = image.convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    generated_text = blip_model.generate(**inputs, max_new_tokens=50)
    description = blip_processor.tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return description

# ✅ 让 MedLLaMA 2 进行医学影像分析
def analyze_medical_text(text_input):
    prompt = f"Patient's Medical Image Analysis:\n{text_input}\n\nProvide a detailed medical diagnosis and possible recommendations."
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
    output = llama_model.generate(**inputs, max_new_tokens=200)
    diagnosis = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    return diagnosis

# ✅ 处理用户上传的 X-ray 图片
def process_image(image):
    description = generate_image_description(image)  # 影像描述
    diagnosis = analyze_medical_text(description)  # 影像诊断
    return description, diagnosis

# ✅ 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 AI 医学影像辅助诊断 (BLIP-2 + MedLLaMA 2)")
    
    with gr.Row():
        image_input = gr.Image(label="上传医学影像（如 X-ray）", type="pil")
        image_output = gr.Label(label="自动生成的影像描述")

    diagnosis_output = gr.Textbox(label="AI 生成的医学诊断", lines=5)
    
    analyze_button = gr.Button("⚡ 进行 AI 诊断")
    
    analyze_button.click(fn=process_image, inputs=image_input, outputs=[image_output, diagnosis_output])

# ✅ 运行 Web 应用
if __name__ == "__main__":
    demo.launch()
