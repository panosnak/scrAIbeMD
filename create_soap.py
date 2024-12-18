import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

def read_patient_info(file_path):
    """
    Reads patient information from a .txt file.

    Args:
        file_path (str): Path to the input .txt file.

    Returns:
        str: Contents of the file as a string.
    """
    return Path(file_path).read_text()

def generate_soap_note(transcribed_audio, model_id="meta-llama/Llama-2-7b-chat-hf"):
    """
    Generates a SOAP note based on the provided patient transcription.

    Args:
        transcribed_audio (str): Details about the patient (symptoms, observations, etc.).

    Returns:
        str: Generated SOAP note.
    """
    # Set up device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the LLaMA model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    prompt = f"Generate a detailed SOAP note for the following patient case based on the transcribed consultation below:\n{transcribed_audio}\n\nSOAP Note:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            temperature=0.1,
            top_p=0.95,
            do_sample = False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Example patient information
# patient_case = """
# The patient is a 45-year-old male presenting with a persistent cough and fever for the past 4 days.
# He reports mild chest pain during coughing. No shortness of breath. No known allergies.
# History of mild asthma. Vital signs: Temp: 101°F, HR: 88 bpm, BP: 130/85 mmHg.
# """

# # # Generate SOAP note
# soap_note = generate_soap_note()
# print(soap_note)
