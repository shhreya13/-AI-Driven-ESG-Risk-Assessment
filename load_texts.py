import os

def load_cleaned_texts(path="cleaned_reports"):
    """
    Load all cleaned company text reports from the given folder.
    Returns a dictionary: {company_name: text_content}.
    """
    company_texts = {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Folder not found: {path}. Please create it and add .txt reports.")

    for fname in os.listdir(path):
        if fname.endswith(".txt"):
            try:
                with open(os.path.join(path, fname), "r", encoding="utf-8", errors="ignore") as f:
                    company_name = os.path.splitext(fname)[0]
                    company_texts[company_name] = f.read()
            except Exception as e:
                print(f"⚠️ Error reading {fname}: {e}")

    if not company_texts:
        print("⚠️ No .txt files found in the folder. Please add company reports.")

    return company_texts
