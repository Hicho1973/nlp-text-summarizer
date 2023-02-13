
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import os

# --- Configuration --- #
MODEL_NAME = "sshleifer/distilbart-cnn-12-6" # A good general-purpose summarization model
MAX_INPUT_LENGTH = 1024
MIN_SUMMARY_LENGTH = 30
MAX_SUMMARY_LENGTH = 150

# --- 1. Initialize Summarization Pipeline --- #
def initialize_summarizer(model_name=MODEL_NAME):
    """Initializes a Hugging Face summarization pipeline."""
    print(f"Initializing summarization pipeline with model: {model_name}")
    try:
        summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
        print("Summarization pipeline initialized successfully.")
        return summarizer
    except Exception as e:
        print(f"Error initializing summarizer: {e}")
        print("Attempting to load model and tokenizer manually.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        print("Summarization pipeline initialized successfully (manual load).")
        return summarizer

# --- 2. Perform Summarization --- #
def summarize_text(summarizer, text, min_length=MIN_SUMMARY_LENGTH, max_length=MAX_SUMMARY_LENGTH):
    """Performs abstractive summarization on the given text."""
    if not text or len(text.strip()) == 0:
        return ""
    
    print(f"Summarizing text (length: {len(text)})...")
    try:
        summary = summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print(f"Error during summarization: {e}")
        return ""

# --- 3. Batch Summarization Example --- #
def batch_summarize_dataset(summarizer, dataset_name="cnn_dailymail", split="test", num_samples=5):
    """Demonstrates batch summarization on a dataset."""
    print(f"Loading dataset for batch summarization: {dataset_name} ({split} split)...")
    try:
        dataset = load_dataset(dataset_name, "3.0.0", split=split)
        # Take a small sample for demonstration
        sample_texts = [item["article"] for item in dataset.shuffle(seed=42).select(range(num_samples))]
        
        print(f"Summarizing {num_samples} articles...")
        summaries = []
        for i, text in enumerate(sample_texts):
            print(f"  - Summarizing article {i+1}/{num_samples}...")
            summaries.append(summarize_text(summarizer, text))
        
        print("\n--- Batch Summarization Results ---")
        for i, (original, summary) in enumerate(zip(sample_texts, summaries)):
            print(f"\nArticle {i+1}:\nOriginal: {original[:500]}...\nSummary: {summary}")
        return summaries
    except Exception as e:
        print(f"Error during batch summarization: {e}")
        return []

# --- Main Execution --- #
if __name__ == "__main__":
    summarizer_pipeline = initialize_summarizer()
    
    # Example 1: Summarize a single piece of text
    sample_article = (
        "The Orbiter Discovery, commanded by veteran astronaut Mark Polansky, "
        "launched on schedule from Kennedy Space Center at 11:38 a.m. EST. "
        "The mission, designated STS-116, is a critical step in the ongoing "
        "construction of the International Space Station (ISS). Discovery "
        "is carrying a new segment of the station's truss, which will expand "
        "its power and cooling capabilities. The seven-member crew will also "
        "perform three spacewalks to reconfigure the station's power grid "
        "and prepare for future additions. This mission marks the first time "
        "a space shuttle has launched at night since 2002, providing a "
        "spectacular view for spectators along the Florida coast. The shuttle "
        "is expected to dock with the ISS on Sunday, where the crew will be "
        "greeted by the station's current residents. The entire mission is "
        "expected to last 12 days, with Discovery returning to Earth on "
        "December 21st." 
    )
    print("\n--- Single Text Summarization ---")
    single_summary = summarize_text(summarizer_pipeline, sample_article)
    print(f"Original: {sample_article}\nSummary: {single_summary}")
    
    # Example 2: Batch summarization from a dataset
    # Note: This requires downloading the 'cnn_dailymail' dataset, which can be large.
    # It's commented out by default to avoid long download times during initial setup.
    # Uncomment the line below to run batch summarization.
    # batch_summarize_dataset(summarizer_pipeline)
