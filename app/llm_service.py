# required dependencies
import os, json
import re
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model

# cred
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

# initializing the model
modelgroq=init_chat_model("groq:qwen/qwen3-32b")
# print(modelgroq)
print("Model initialised successfully.")


def get_llm_response(disease_name: str, confidence_score: float):

    prompt = f"""
    You are an expert agricultural assistant specializing in plant diseases.

    INPUT:
    Detected Disease: {disease_name}
    Confidence Score: {confidence_score}

    Return ONLY valid JSON:
    {{
        "disease_confirmation": "",
        "description": "",
        "organic_treatment": [],
        "chemical_treatment": [],
        "prevention": [],
        "note": ""
    }}
    """

    response = modelgroq.invoke(prompt)

    # remove <think> tags
    clean_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()

    print("RAW:", repr(clean_content))  # DEBUG

    match = re.search(r"\{.*\}", clean_content, re.DOTALL)
    if match:
        clean_content = match.group(0)
    else:
        raise ValueError("No JSON found in response")

    try:
        response_text = json.loads(clean_content)
    except json.JSONDecodeError as e:
        print("JSON ERROR:", e)
        print("CONTENT:", clean_content)
        raise

    return response_text   
