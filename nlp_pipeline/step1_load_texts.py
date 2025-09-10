import os

# Folder where Person 1 sent the cleaned text files
folder = "cleaned_reports/"  # Make sure this folder exists and contains the .txt files

# List all .txt files in the folder
files = [f for f in os.listdir(folder) if f.endswith(".txt")]

# Dictionary to store company name → cleaned text
company_texts = {}

for file in files:
    # Remove .txt extension to use as company name
    company_name = file.replace(".txt", "")
    
    # Open and read the text file
    with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
        text = f.read().lower()  # lowercase for easier NLP matching
        company_texts[company_name] = text

# Optional: preview the loaded texts
for company, text in company_texts.items():
    print(f"{company}: {len(text)} characters loaded")
