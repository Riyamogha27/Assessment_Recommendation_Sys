# SHL-Assessment-Recommendation-System

This repository contains a Generative AI-powered recommendation system designed to suggest suitable SHL assessments based on natural language queries or job descriptions. The tool leverages LLMs and NLP techniques to match user input to the most relevant SHL assessment tests.

> ⚠️ Note: This application **requires GPU** support for optimal performance and is **not deployable on Streamlit Cloud** due to hardware limitations.

## 🚀 Overview

The project performs the following:

1. Scrapes and preprocesses data from the SHL website for both Pre-Packaged and Individual Test Solutions.
2. Uses NLP and embedding-based semantic similarity to match user input to assessment descriptions.
3. Provides a user-friendly interface via Streamlit to input queries and get recommendations.
4. Optionally, supports deep learning models (e.g., Sentence Transformers) for advanced matching.

## 🛠️ Installation

1. **Create a Python environment:**

    ```bash
    conda create -n shl-ai python=3.10
    conda activate shl-ai
    ```

2. **Install PyTorch (GPU-enabled):**

   - **Linux:**
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - **For other systems:** Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and follow GPU install instructions.

3. **Install project dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app (locally with GPU):**

    ```bash
    streamlit run app.py
    ```

## 💻 Usage

Once the app is running:

- Input a **natural language job description** or **skills profile**.
- The app will process the input and return **top-matching SHL assessments**.
- You can review details like assessment name, duration, and skills evaluated.

## 📁 Project Structure

- `app.py` – Main Streamlit application.
- `utils/` – Utility scripts for scraping, processing, and matching.
- `data/` – Raw and cleaned SHL assessment data (JSON/CSV formats).
- `models/` – Pre-trained NLP models or saved embeddings (optional).
- `requirements.txt` – Python dependencies.

## ⚙️ Model Support

- [x] Sentence Transformers (`all-MiniLM-L6-v2`, `multi-qa-MiniLM`)
- [x] TF-IDF + cosine similarity (fallback for lighter environments)
- [ ] OpenAI/GPT API support (optional for enhanced embeddings)

## ⚠️ Limitations

- Deployment to **Streamlit Cloud is not supported** due to lack of GPU support.
- Embedding models require **moderate to high GPU memory** (>=4GB recommended).
- For public deployment, consider **AWS EC2 (GPU), Google Colab**, or **Hugging Face Spaces with GPU**.

## 📌 Future Enhancements

- Add support for Chat-based interaction.
- Integration with SHL API (if publicly available).
- Add visual analytics for assessment clustering.
