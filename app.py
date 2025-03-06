import gradio as gr
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

# ✅ Set device (use GPU acceleration if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load the BLIP-2 vision model
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float32).to(device)

# ✅ Load the MedLLaMA 2 language model
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float32, device_map="auto")

# ✅ Process medical images
def generate_image_description(image):
    image = image.convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    generated_text = blip_model.generate(**inputs, max_new_tokens=50)
    description = blip_processor.tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return description

# ✅ Use MedLLaMA 2 to analyze medical images
def analyze_medical_text(text_input):
    prompt = f"Patient's Medical Image Analysis:\n{text_input}\n\nProvide a detailed medical diagnosis and possible recommendations."
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
    output = llama_model.generate(**inputs, max_new_tokens=200)
    diagnosis = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    return diagnosis

# ✅ Process user-uploaded X-ray images
def process_image(image):
    description = generate_image_description(image)  # Generate image description
    diagnosis = analyze_medical_text(description)  # Generate medical diagnosis
    return description, diagnosis

# ✅ Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 AI-Assisted Medical Image Diagnosis")
    
    with gr.Row():
        image_input = gr.Image(label="Upload medical image (e.g., X-ray)", type="pil")
        image_output = gr.Label(label="Automatically generated image description")

    diagnosis_output = gr.Textbox(label="AI-generated medical diagnosis", lines=5)
    
    analyze_button = gr.Button("⚡ Perform AI Diagnosis")
    
    analyze_button.click(fn=process_image, inputs=image_input, outputs=[image_output, diagnosis_output])

# ✅ Run the web application
if __name__ == "__main__":
    demo.launch(share=True)
