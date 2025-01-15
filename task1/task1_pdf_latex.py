from dotenv import load_dotenv
import os 
import base64
import google.generativeai as genai
from edsl import QuestionNumerical, Model
from dotenv import load_dotenv
import subprocess
import select
import sys
from math import factorial, exp, log
import numpy as np 
import pandas as pd


# change to location of file
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
# change to ~/PythonPackages/wildbench/task1
os.chdir(os.getenv("HOME") + "/PythonPackages/wildbench/task1")
# Setup
load_dotenv("../.env", verbose=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
pdf_model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key=GOOGLE_API_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt4o = Model("gpt-4o")
edsl_models = [gpt4o]


def compile_latex(tex_file, output_dir="."):
    try:
        print("Attempting to compile LaTeX file...")
        process = subprocess.Popen(
            ["pdflatex", "-output-directory", output_dir, tex_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True
        )
        
        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [])

            for fd in ret[0]:
                if fd == process.stdout.fileno():
                    line = process.stdout.readline()
                    if line:
                        print(line.strip())
                        if "?" in line:  # Detected prompt
                            process.stdin.write("\n")
                            process.stdin.flush()
                elif fd == process.stderr.fileno():
                    line = process.stderr.readline()
                    if line:
                        print(line.strip())
            
            if process.poll() is not None:
                break
        
        if process.returncode == 0:
            print(f"Compilation successful. Output in: {output_dir}")
        else:
            print("Compilation completed with warnings")
            
    except Exception as e:
        print(f"Error in compilation: {str(e)}")

def c_l(gamma_l, a_1, tau_l):
        return (1 - gamma_l) / (a_1 + tau_l)

def V_l(a_l, beta, X_l, gamma_l, tau_l):
    return beta * X_l + log(gamma_l) + (gamma_l - 1) * log(a_l + tau_l)

L = 10
K = 3
a = np.array([10, 1, 3])
a = np.concatenate([a, np.zeros(L - K)])
gamma = np.array([0.5] * L)
tau = np.array([0] + [1] * (L - 1))
beta = 2
X = np.array([3] * L)
c = np.array([c_l(gamma[l], a[0], tau[l]) for l in range(L)])
V = np.array([V_l(a[l], beta, X[l], gamma[l], tau[l]) for l in range(L)])

def model_LL(a, K, L, c, V):
    term1 = np.prod(c[0:K])
    term2 = sum(1 / np.array(c[0:K]))
    term3_num = np.prod(np.exp(V[0:K]))
    term3_denom = sum(np.exp(V))**K
    term3 = term3_num / term3_denom
    term4 = factorial(K - 1)
    return log(term1 * term2 * term3 * term4)

true_LL = model_LL(a, K, L, c, V)


repetitions = 10
successes = np.zeros(repetitions)
for rep in range(repetitions):
    # Step 1: Convert a PDF page to LaTeX
    doc_path = "equation_page.pdf" # Replace with the actual path to your local PDF
    with open(doc_path, "rb") as doc_file:
        doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")
    prompt = "make LaTeX to reproduce this PDF page."
    response = pdf_model.generate_content([{'mime_type': 'application/pdf', 'data': doc_data}, prompt])

    # Save the LaTeX content to a file
    tex_file_path = "document.tex"
    with open(tex_file_path, "w") as tex_file:
        tex_file.write(response.text)

    tex_file_path = "document.tex"  # Replace with your .tex file path
    compile_latex(tex_file_path)

    # Step 2: Compute a value using the compiled PDF 
    doc_path = "document.pdf" # Replace with the actual path to your local PDF
    with open(doc_path, "rb") as doc_file:
        doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")

    prompt = """Using the equation in this PDF, get the model likelihood for the following values:

    L=10 
    K=3
    a^*_1 = 10, a^*_2 = 1, a^*_3 = 3
    gamma_l = 0.5 for all l
    tau_1 = 0, and tau_l = 1 otherwise 
    beta = 2
    X_l = 3 for all l.

    Please then take the log to provide the final answer (model log-likelihood). Your entire response should be a single number.
    """

    response = pdf_model.generate_content([{'mime_type': 'application/pdf', 'data': doc_data}, prompt])
    print(response.text, "\n\n")

    try:
        computed_LL = float(response.text.strip())
        print(f"True LL: {true_LL}")
        print(f"Computed LL: {computed_LL}")
        if abs(true_LL - computed_LL) < 1:
            successes[rep] = 1
    except:
        question_text = f"Please give the value of the log-likelihood provided in the following text: {response.text}"
        q_LL_extract = QuestionNumerical(question_name = "LL_extract", question_text = question_text)
        print(f"EDSL key in environment? os.getenv('EXPECTED_PARROT_API_KEY') = {os.getenv('EXPECTED_PARROT_API_KEY')}")
        r_LL_extract = q_LL_extract.by(edsl_models).run() #run(disable_remote_inference=True)
        computed_LL = r_LL_extract.select("LL_extract").to_list()[0]
        if abs(true_LL - computed_LL) < 1:
            successes[rep] = 1


print(f"Success rate: {np.mean(successes)}")

def write_success_rate(success_rate, row_name, csv_path='results.csv'):
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame()
    
    if row_name not in df.index:
        df.loc[row_name] = np.nan
    if 'score' not in df.columns:
        df['score'] = np.nan
    
    df.loc[row_name, 'score'] = success_rate
    df.to_csv(csv_path)

write_success_rate(np.mean(successes), 'Task1', csv_path="../results.csv")

