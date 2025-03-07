# biostat-285-final-proj

This project is designed to run on **Lambda Labs GPU instances**. It includes **AI-assisted medical image diagnosis** using **BLIP-2 + MedLLaMA 2** and a **medical chatbot** powered by **BioMistral-7B**.

---

## **1. Connect to the Lambda Labs Server**

Before running the project, **SSH into your Lambda Labs instance**:

```bash
ssh -i ~/.ssh/id_rsa ubuntu@YOUR_INSTANCE_IP
```

## **2. Install Dependencies**

### Check Python Environment

Ensure that you are using the correct Python version (Python 3.10 or higher):

```bash
python --version
```

### Install Required Packages

Run the following command to install all required dependencies:

```bash
pip install -r requirements.txt
```

If pip install fails due to version conflicts, try:

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

## **3. Login to Hugging Face**

The project uses Llama 2 and BioMistral-7B, which are gated models on Hugging Face. You must authenticate before running the app.

### Step 1: Request Model Access

Go to Hugging Face - Llama 2 and request access to the model.

### Step 2: Login to Hugging Face on the Server

Run:

```bash
pip install huggingface_hub  # If not installed
huggingface-cli login
```

Enter your Hugging Face access token when prompted.
You can generate a token from: Hugging Face Tokens

If authentication is successful, your credentials will be saved for future model downloads.

## **4. Run the Application**

Start the Web Interface
Once all dependencies are installed, start the app:

```bash
python app.py
```

If everything is set up correctly, you should see an output like:

```csharp
* Running on local URL: http://127.0.0.1:7860
* Running on public URL: https://xxxxx.gradio.live
```

Click on the public Gradio link to access the application from your browser.
