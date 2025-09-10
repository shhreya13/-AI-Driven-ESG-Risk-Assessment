import os
import pandas as pd
from textblob import TextBlob
from risk_terms import risk_terms, weight_map

# Folder containing cleaned .txt reports
INPUT_FOLDER = "cleaned_reports"
OUTPUT_FILE = "esg_risk_output.csv"


def read_file(path):
    """
    Read text file with fallback encodings to avoid decode errors.
    """
    encodings = ["utf-8", "latin-1", "ISO-8859-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    # Last resort: ignore bad characters
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_risks(text, company):
    """
    Extract ESG risk terms, counts, sentiment, and weighted scores from company text.
    """
    results = []
    text_lower = text.lower()

    for category, terms in risk_terms.items():
        for term in terms:
            count = text_lower.count(term.lower())
            if count > 0:
                # Get sentences containing the term
                sentences = [s for s in text.split(".") if term in s]
                sentiment = (
                    sum(TextBlob(s).sentiment.polarity for s in sentences) / len(sentences)
                    if sentences else 0
                )
                risk_weight = weight_map.get(term, 1)
                weighted_score = count * risk_weight

                results.append({
                    "company": company,
                    "category": category,
                    "term": term,
                    "count": count,
                    "sentiment": round(sentiment, 2),
                    "risk_weight": risk_weight,
                    "weighted_score": weighted_score,
                })

    return results


def main():
    all_rows = []

    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        return

    files = os.listdir(INPUT_FOLDER)
    if not files:
        print("⚠️ No files found in input folder.")
        return

    for file in files:
        if file.endswith(".txt"):
            company = os.path.splitext(file)[0]
            path = os.path.join(INPUT_FOLDER, file)
            try:
                text = read_file(path)
                rows = extract_risks(text, company)
                all_rows.extend(rows)
                print(f"✅ Processed {company} ({len(rows)} risk terms found)")
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ ESG risk extraction complete. Results saved in {OUTPUT_FILE}")
    else:
        print("⚠️ No risk terms found in any reports.")


if __name__ == "__main__":
    main()
