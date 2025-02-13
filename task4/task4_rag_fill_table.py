from dotenv import load_dotenv
import os 
import base64
from glob import glob
from edsl import QuestionFreeText, Model
import numpy as np 
import pandas as pd
from openai import OpenAI


client = OpenAI()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Setup
load_dotenv("../.env", verbose=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt4o = Model("gpt-4o")
edsl_models = [gpt4o]


paths = glob("*.txt")
with open(paths[0], "r") as file:
    text_one = file.read()
with open(paths[1], "r") as file:
    text_two = file.read()

table_true = pd.read_csv("supplement_table.csv")
true_rows = table_true["Supplement"].tolist()
true_rows = [row for row in true_rows if str(row) != 'nan']


repetitions = 10
score = np.zeros(repetitions)
for rep in range(repetitions):
        
    message1 = [{
        "role": "user",
        "content": f"""I want you to make a single column table from the following text. The rows
        should have supplement names, the column is health category of the document, and the cells are
        evidence categories: Primary, Secondary, Promising, Unproven, and Inadvisable. The text is:
        {text_one}"""
    }]

    completion = client.chat.completions.create(
        model="gpt-4o", 
        messages=message1
    )

    r1 = completion.choices[0].message.content

    message2 = [{
        "role": "user",
        "content": f"""I want you to make a single column table from the following text. The rows
        should have supplement names, the column is health category of the document, and the cells are
        evidence categories: Primary, Secondary, Promising, Unproven, and Inadvisable. The text is:
        {text_two}"""
    }]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message2
    )

    r2 = completion.choices[0].message.content

    message3 = [{
        "role": "user",
        "content": f"""I want you to construct a table from the following two columns: {r1} and
        {r2}. The rows should be the supplement names, the columns should be the health categories, and
        the cells should be the evidence categories. Please format the table as a CSV file."""
    }]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message3
    )
    r3 = completion.choices[0].message.content


    extract_csv_text = f"""Please extract the information from the response that can be formatted as a
        CSV file, NOT INCLUDING any tags such as ```csv```: {r3}"""
    q_extract_csv = QuestionFreeText(question_name = "extract_csv", question_text = extract_csv_text)
    extract_csv = q_extract_csv.by(edsl_models).run(disable_remote_inference=True)
    extract_csv.select("extract_csv").print(format="rich")
    csv_path = "extracted_table.csv"
    with open(csv_path, "w") as csv_file:
        csv_string = extract_csv.select("extract_csv")[0]["answer.extract_csv"][0]
        csv_file.write(csv_string)

    try:
        table_model = pd.read_csv("extracted_table.csv", index_col=0)
        table_model.index = table_model.index.str.lower().str.strip()
        extracted_rows = table_model.index.tolist()
        extracted_rows = [row for row in extracted_rows if str(row) != 'nan']

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
            if table_true.loc[row].isnull().values.any():
                true_row = table_true.loc[row].tolist() 
                extracted_row = table_model.loc[row].tolist()
                true_nans = sum([1 for i in range(len(true_row)) if str(true_row[i]) == 'nan'])
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

write_success_rate(np.mean(score), 'Task4', csv_path="../results.csv")
