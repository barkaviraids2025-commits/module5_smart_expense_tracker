"""Quick checks for the AI parts. Run:  python test_expense_ai.py"""

from datetime import date

from expense_ai import forecast_next_month, suggest_category, train_categorizer

# Titles the model has NEVER seen in its training data
UNSEEN = {
    "evening tea": "Food", "dosa": "Food", "pizza": "Food", "fresh fruits": "Food",
    "bus fare": "Travel", "taxi": "Travel", "petrol bunk": "Travel", "train booking": "Travel",
    "new shoes": "Shopping", "amazon purchase": "Shopping", "earbuds": "Shopping", "tshirt": "Shopping",
    "current bill": "Bills", "electric bill": "Bills", "broadband": "Bills", "recharge": "Bills",
    "pen and notebook": "Education", "exam fees": "Education", "book purchase": "Education", "printout": "Education",
    "medicines": "Other", "movie": "Other", "donation": "Other", "laundry": "Other",
}


def test_categoriser():
    model = train_categorizer()
    correct = 0
    for title, expected in UNSEEN.items():
        guess, _ = suggest_category(model, title)
        correct += guess == expected
        if guess != expected:
            print(f"  miss: '{title}' -> {guess} (expected {expected})")
    accuracy = correct / len(UNSEEN)
    print(f"Categoriser accuracy on unseen titles: {correct}/{len(UNSEEN)} = {accuracy:.0%}")
    assert accuracy >= 0.75


def test_forecast():
    rows = [{"date": f"2026-0{m}-10", "amount": 1000 + 100 * m, "title": "x", "category": "Food"} for m in range(1, 6)]
    result = forecast_next_month(rows, today=date(2026, 6, 15))
    print(f"Forecast for a steadily rising trend: Rs {result['amount']:.0f} ({result['method']})")
    assert result["method"] == "trend"
    assert abs(result["amount"] - 1600) < 1  # perfect upward line: 1100..1500 -> next 1600
    assert forecast_next_month([], today=date(2026, 6, 15)) is None


if __name__ == "__main__":
    test_categoriser()
    test_forecast()
    print("All checks passed.")
