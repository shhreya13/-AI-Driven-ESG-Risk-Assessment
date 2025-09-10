import os
import pandas as pd
from transformers import pipeline
import spacy
from risk_terms import risk_terms, weight_map
from load_texts import load_cleaned_texts


# ---------------- Setup ----------------
OUTPUT_FILE = "esg_risk_output.csv"

# Hugging Face sentiment model (DistilBERT fine-tuned on SST-2)
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# spaCy NER model
nlp = spacy.load("en_core_web_sm")


# ---------------- Helpers ----------------
def analyze_sentiment(sentences):
    """
    Use Hugging Face transformer to get sentiment polarity for a list of sentences.
    Positive → +score, Negative → -score
    """
    sentiments = []
    for s in sentences:
        try:
            res = sentiment_model(s[:512])[0]  # limit to 512 tokens
            polarity = res["score"] if res["label"] == "POSITIVE" else -res["score"]
            sentiments.append(polarity)
        except Exception:
            sentiments.append(0)
    return sum(sentiments) / len(sentiments) if sentiments else 0


def extract_risks(company, text):
    """
    Extract ESG risks from text:
    - Lexicon-based keyword detection
    - Transformer-based sentiment
    - spaCy NER for entities (e.g., regulators, laws, countries)
    """
    results = []
    text_lower = text.lower()

    # --- ESG Keyword Search ---
    for category, terms in risk_terms.items():
        for term in terms:
            count = text_lower.count(term.lower())
            if count > 0:
                # Find sentences with the term
                sentences = [s.strip() for s in text.split(".") if term in s]
                sentiment = analyze_sentiment(sentences) if sentences else 0

                risk_weight = weight_map.get(term, 1)
                weighted_score = count * risk_weight

                results.append({
                    "company": company,
                    "category": category,
                    "term": term,
                    "count": count,
                    "sentiment": round(sentiment, 2),
                    "risk_weight": risk_weight,
                    "weighted_score": weighted_score
                })

    # --- Entity Detection (Regulators, Laws, etc.) ---
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "LAW", "GPE"]:  # SEC, EPA, EU, Supreme Court, etc.
            results.append({
                "company": company,
                "category": "Governance",
                "term": f"Entity: {ent.text}",
                "count": 1,
                "sentiment": 0,
                "risk_weight": 1,
                "weighted_score": 1
            })

    return results


# ---------------- Main ----------------
def main():
    company_texts = load_cleaned_texts()
    all_results = []

    for company, text in company_texts.items():
        try:
            rows = extract_risks(company, text)
            all_results.extend(rows)
            print(f"✅ Processed {company}: {len(rows)} risks flagged")
        except Exception as e:
            print(f"❌ Error processing {company}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ ESG risk extraction complete. Results saved to {OUTPUT_FILE}")
    else:
        print("⚠️ No risks detected in any company reports.")


if __name__ == "__main__":
    main()
