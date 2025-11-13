# ==============================================================================
# CELL 2: THE FINAL APPLICATION - USING THE W&B MODEL
# ==============================================================================
import chromadb
import shutil
import os
from dotenv import load_dotenv
load_dotenv()
from .retrive import search
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import InferenceClient


# Run this only if model_dir was successfully set in the previous cell:


    # !!! IMPORTANT: Update this path to your ChromaDB folder on Google Drive !!!


# --- Load your fine-tuned BART model (the 'Chef') from the W&B directory ---


# --- The ChromaDB Retriever Class ---

# --- The Interactive Tutor Function with RAG ---
def run_tutor_with_rag(topic):
    
    db_local_path = "https://huggingface.co/datasets/Shivani4444/mlops-ragsystem-chroma/resolve/main"
    model_id = "chershilhyde/bart-qna-generator-finetuned"
    print(f"\n🔍 Attempting to load fine-tuned BART model from Hugging Face: {model_id}")
    
    hf_token = os.getenv("HF_API_TOKEN")  
    
    try:
        print("→ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        print("✅ Tokenizer loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error while loading tokenizer: {e}")
        return
    
    try:
        print("→ Loading model weights...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=hf_token)
        model.eval()
        print("✅ Model weights loaded successfully and ready for inference.")
    except Exception as e:
        print(f"⚠️ Error while loading model: {e}")
        return
    
    
    
    
    
    print("=" * 60)
    print(f"TUTOR: You asked about '{topic}'.")

    print("  -> Searching for relevant context in ChromaDB...")
    context = search(topic)

    if not context:
        print("TUTOR: I couldn't find any information on that topic.")
        return
    print("  -> Context found! Generating a question...")

    prompt = f"generate question and answer: ```{context}```"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    output_ids = model.generate(inputs.input_ids, max_length=256, num_beams=5, early_stopping=True)
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)


    try:
        question_part, answer_part = decoded_output.split("answer:")
        generated_question = question_part.replace("question:", "").strip()
        correct_answer = answer_part.strip()
        
        
    except ValueError:
        print("TUTOR: I found information but had trouble forming a question.")
        return
    
    print("from bart model")
    print(f"\nGenerated Question: {generated_question}")
    print(f"Correct Answer: {correct_answer}"  )
    print("  -> Refining the question and answer using Llama 3...")       
    
    prompt = f"""
    Refine the question and answer below. 
    Do NOT add new information. 
    Do NOT shorten too much.
    Keep meaning the same. Make wording clear and natural.

    Question: {question_part}
    Answer: {answer_part}

    Return format strictly:
    Question: ...
    Answer: ...
    """
    
    
    # api_key = os.getenv("GEMINI_API_KEY")
    # client = genai.Client(api_key=api_key)
    # response = client.models.generate_content(
    # model="gemini-2.5-flash",  
    # contents=prompt)
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    client = InferenceClient(model_name, token=os.getenv("HF_API_TOKEN"))
    
    try:
        res =client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )
        response=res.choices[0].message["content"]
        print("---------------------------------")
        print(response)
        print("---------------------------------")
    except Exception as e:
        return f"⚠️ Error communicating with model: {e}"
    
    ref_q, ref_a = "", ""
    refined_question, refined_correct_answer = "", ""
    
    try:
        ref_q, ref_a = response.split("Answer:")
        refined_question = ref_q.replace("Question:", "").strip()
        refined_correct_answer = ref_a.strip()
    except:
        refined_question = generated_question
        refined_correct_answer = correct_answer
    
    
    
    print(f"\nQUESTION: {refined_question}")
    # user_answer = input("YOUR ANSWER: ")

    # print("-" * 25)
    # print(f"Your Answer:    '{user_answer}'")
    # print(f"Correct Answer: '{refined_correct_answer}'")
    # print("=" * 60)
    
    return refined_correct_answer,refined_question
    
    
    
    
    # The 'contents' argument is preferred)

# --- Let's run it! ---




#run_tutor_with_rag("carbon")