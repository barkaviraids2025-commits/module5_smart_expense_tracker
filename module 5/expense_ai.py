"""
AI helpers for the Smart Expense Tracker.

1. Auto-categoriser : text classifier (TF-IDF + Logistic Regression) that guesses
                      the category from the expense title.
2. Spending forecast: linear regression on past monthly totals to predict next month.
"""

from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

CATEGORIES = ["Food", "Travel", "Shopping", "Bills", "Education", "Other"]

# Starter examples so the model works from day one
TRAINING_DATA = {
    "Food": ["lunch at canteen", "biryani", "tea and snacks", "dinner with friends", "swiggy order",
             "zomato pizza", "coffee", "fruit juice", "groceries", "vegetables", "milk packet",
             "breakfast idli", "ice cream", "bakery cake", "chicken rice", "parotta", "samosa",
             "chocolate"],
    "Travel": ["bus ticket", "auto fare", "metro card recharge", "petrol", "train ticket", "uber ride",
               "ola cab", "bike fuel", "flight ticket", "parking fee", "toll charge", "cab to airport",
               "college bus fee", "rapido", "bus pass"],
    "Shopping": ["t-shirt", "jeans", "shoes", "earphones", "phone cover", "backpack", "watch",
                 "amazon order", "flipkart purchase", "sandals", "dress", "bag", "sunglasses", "makeup",
                 "kurta", "headphones"],
    "Bills": ["electricity bill", "EB bill", "mobile recharge", "wifi bill", "water bill", "gas cylinder",
              "DTH recharge", "internet bill", "rent", "phone bill", "insurance premium",
              "netflix subscription", "spotify subscription", "maintenance charge"],
    "Education": ["notebook", "pen set", "python book", "exam fee", "course fee", "udemy course", "xerox",
                  "project printout", "lab record", "college fees", "coding test fee", "stationery",
                  "textbook", "certification exam", "library fine", "online course"],
    "Other": ["haircut", "doctor consultation", "medicine", "donation", "temple offering",
              "gift for friend", "repair", "laundry", "movie ticket", "charity", "tailor",
              "bank charges", "pet food", "salon", "birthday party"],
}


def train_categorizer(saved_expenses=None):
    """Train on the starter examples plus the user's own saved expenses (so it learns over time)."""
    texts, labels = [], []
    for category, titles in TRAINING_DATA.items():
        texts += titles
        labels += [category] * len(titles)
    for e in saved_expenses or []:
        texts.append(e["title"])
        labels.append(e["category"])

    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True),
        LogisticRegression(max_iter=1000, C=10),
    )
    model.fit(texts, labels)
    return model


def suggest_category(model, title):
    """Return (best_category, confidence between 0 and 1)."""
    probs = model.predict_proba([title])[0]
    best = int(np.argmax(probs))
    return model.classes_[best], float(probs[best])


def monthly_totals(expenses):
    """{'2026-09': 4200.0, '2026-10': 4650.0, ...} sorted by month."""
    totals = {}
    for e in expenses:
        month = e["date"][:7]
        totals[month] = totals.get(month, 0) + e["amount"]
    return dict(sorted(totals.items()))


def forecast_next_month(expenses, today=None):
    """
    Predict next month's total spending from past COMPLETE months.
    3+ months of history -> linear regression trend, otherwise a simple average.
    """
    today = today or date.today()
    this_month = today.strftime("%Y-%m")
    past = {m: t for m, t in monthly_totals(expenses).items() if m < this_month}
    if not past:
        return None

    values = list(past.values())
    if len(values) < 3:
        return {"amount": float(np.mean(values)), "method": "average", "months_used": len(values), "change": 0.0}

    x = np.arange(len(values)).reshape(-1, 1)
    reg = LinearRegression().fit(x, values)
    predicted = float(reg.predict([[len(values)]])[0])
    return {"amount": max(predicted, 0.0), "method": "trend", "months_used": len(values), "change": float(reg.coef_[0])}
