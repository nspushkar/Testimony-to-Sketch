# Testimony-to-Sketch
#  Criminal Poster Generation System 

An AI-powered pipeline that generates realistic criminal poster images **from audio descriptions alone**, using cutting-edge models in speech recognition, natural language processing, and generative vision.

<p align="center">
  <img src="generated_image.png" width="400" alt="Generated Mugshot Example">
</p>

---

##  Overview

This project demonstrates an end-to-end intelligent system that aids **law enforcement agencies** by converting **voice descriptions of suspects** into **visual mugshots**.

It leverages:
-  **OpenAI Whisper** for automatic speech transcription
-  **LLaMA 3.2B Instruct** (via HuggingFace) for trait extraction
-  **Stable Diffusion XL (Base + Refiner)** for high-fidelity image generation

---

##  Tech Stack

| Component      | Technology Used                              |
|----------------|-----------------------------------------------|
| Transcription  | [OpenAI Whisper](https://github.com/openai/whisper)       |
| NLP Extraction | [LLaMA 3.2B Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct) |
| Image Generation | [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| Backend        | Python 3.10, PyTorch, Transformers, Diffusers |
| Visualization  | PIL / Image Viewer                           |

---

##  How It Works

1. ** Audio Input:**  
   The system accepts an `.mp3` or `.wav` file containing a verbal description of a suspect.

2. **✍ Speech-to-Text:**  
   The audio is transcribed using **Whisper** with medium accuracy model.

3. ** Attribute Extraction:**  
   The transcribed description is passed to **LLaMA 3.2B** to extract structured features such as:
   - Gender, Age, Build, Race
   - Facial Features, Hair, Eyes, Skin Tone
   - Distinctive Traits, Clothing, etc.

4. ** Image Prompt Generation:**  
   These features are formatted into a natural language prompt for visual synthesis.

5. ** Image Generation:**  
   The prompt is processed using **Stable Diffusion XL**, first via the base model, then refined for higher quality output.

6. ** Output:**  
   The resulting mugshot is saved and displayed automatically.

---

##  Achievements

-  **Winner – CodeMav 2025 Hackathon**
- Recognized for its innovative application of generative AI in public safety and investigation workflows.

---

## 📂 Example Usage

```bash
# Ensure your environment includes:
# whisper, torch, transformers, diffusers, accelerate, PIL

# Step 1: Transcribe audio
python transcribe.py --file path/to/audio.mp3

# Step 2: Extract features
python extract_traits.py --text "transcribed text from step 1"

# Step 3: Generate image
python generate_image.py --prompt "auto-generated prompt from step 2"
