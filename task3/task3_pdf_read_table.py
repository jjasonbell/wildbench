from dotenv import load_dotenv
import os 
import base64
import google.generativeai as genai
from edsl import QuestionFreeText, Model
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
score = np.zeros(repetitions)
for rep in range(repetitions):
    doc_path = "alphabet_10K.pdf" 
    with open(doc_path, "rb") as doc_file:
        doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")

    prompt = "get the table on page 55, the Consolidated Statement of Cash Flows, and reproduce it as a CSV file"
    response = pdf_model.generate_content([{'mime_type': 'application/pdf', 'data': doc_data}, prompt])
    extract_csv_text = f"Please extract the information from the response that can be formatted as a CSV file, NOT INCLUDING any tags such as ```csv```: {response.text}"
    q_extract_csv = QuestionFreeText(question_name = "extract_csv", question_text = extract_csv_text)
    extract_csv = q_extract_csv.by(edsl_models).run(disable_remote_inference=True)
    extract_csv.select("extract_csv").print(format="rich")

    csv_path = "extracted_table.csv"
    with open(csv_path, "w") as csv_file:
        csv_string = extract_csv.select("extract_csv")[0]["answer.extract_csv"][0]
        csv_file.write(csv_string)

    table_true = pd.read_csv("alphabet_10K_cash_flows.csv", index_col=0)
    table_true.index = table_true.index.str.lower().str.strip()
    true_rows = table_true.index.tolist()
    true_rows = [row for row in true_rows if str(row) != 'nan']
    try:
        table_model = pd.read_csv("extracted_table.csv", index_col=0)
        table_model.index = table_model.index.str.lower().str.strip()
        extracted_rows = table_model.index.tolist()
        extracted_rows = [row for row in extracted_rows if str(row) != 'nan']

        # get subset of true rows that exist in the extracted table after stripping whitespace
        def get_true_rows(true_rows, extracted_rows):
            true_rows = [row for row in true_rows if row in extracted_rows]
            return true_rows

        true_rows_matched = get_true_rows(true_rows, extracted_rows)
        prop_true_rows = len(true_rows_matched)/len(true_rows)
        n_cells_matched = []
        total_cells = 0
        for row in true_rows_matched:
            total_cells += len(table_true.loc[row].tolist())
            true_nans = 0
            # first check to see if any cells are nan
            if table_true.loc[row].isnull().values.any():
                # then check how many cells are nan and how many match
                true_row = table_true.loc[row].tolist() 
                extracted_row = table_model.loc[row].tolist()
                # count how many are nan in both
                true_nans = sum([1 for i in range(len(true_row)) if str(true_row[i]) == 'nan'])
            # see how many cells match 
            true_row = table_true.loc[row].tolist()
            extracted_row = table_model.loc[row].tolist()
            num_cells_matched = sum([1 for i in range(len(true_row)) if true_row[i] == extracted_row[i]])
            total_matched = num_cells_matched + true_nans
            n_cells_matched.append(total_matched)

        prop_cells_matched = sum(n_cells_matched)/total_cells
    except:
        prop_true_rows = 0
        prop_cells_matched = 0
    score[rep] = prop_cells_matched



def write_success_rate(score, row_name, csv_path='results.csv'):
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['score'])
    
    df.loc[row_name, 'score'] = score
    df.to_csv(csv_path)

write_success_rate(np.mean(score), 'Task3', csv_path="../results.csv")
