from dotenv import load_dotenv
import os 
import base64
import google.generativeai as genai
from edsl import QuestionNumerical, Model
from dotenv import load_dotenv
import numpy as np 
import pandas as pd


os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Setup
load_dotenv("../.env", verbose=True)
pdf_model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt4o = Model("gpt-4o")
edsl_models = [gpt4o]

repetitions = 10
successes = np.zeros(repetitions)
for rep in range(repetitions):
    # Step 1: Convert a PDF page to LaTeX
    doc_path = "target_doc.pdf" 
    with open(doc_path, "rb") as doc_file:
        doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")
    prompt = "Count the words in the abstract of the PDF document, and then count the words in the remainder. Provide both counts separately."
    response = pdf_model.generate_content([{'mime_type': 'application/pdf', 'data': doc_data}, prompt])
    print(response.text)
    true_abstract_WC = 374
    true_remainder_WC = 3253
    abstract_q_text = f"Please give the value of the abstract WC provided in the following text: {response.text}"
    q_abstract = QuestionNumerical(question_name = "abstract_WC", question_text = abstract_q_text)
    q_abstract = q_abstract.by(edsl_models).run(disable_remote_inference=True)
    model_abstract_WC = q_abstract.select("abstract_WC").to_list()[0]
    remainder_q_text = f"Please give the value of the remainder WC provided in the following text: {response.text}"
    q_remainder = QuestionNumerical(question_name = "remainder_WC", question_text = remainder_q_text)
    q_remainder = q_remainder.by(edsl_models).run(disable_remote_inference=True)
    model_remainder_WC = q_remainder.select("remainder_WC").to_list()[0]
    # check if both word counts are within 2% of the true values
    if abs(true_abstract_WC - model_abstract_WC) < 0.02*true_abstract_WC and abs(true_remainder_WC - model_remainder_WC) < 0.02*true_remainder_WC:
        successes[rep] = 1


print(f"Success rate: {np.mean(successes)}")

def write_success_rate(success_rate, row_name, csv_path='results.csv'):
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['score'])
    
    df.loc[row_name, 'score'] = success_rate
    df.to_csv(csv_path)

write_success_rate(np.mean(successes), 'Task2', csv_path="../results.csv")