"""
Smart Expense Tracker - Module 05 (AI Tools & Mini Project)
An upgrade of my Module 02 Expense Tracker with two AI features:
  * auto-suggests the category from the expense title
  * forecasts next month's spending
"""

import json
import random
from datetime import date

from expense_ai import CATEGORIES, forecast_next_month, suggest_category, train_categorizer

FILE_NAME = "expenses.json"


# ---------- saving and loading ----------
def load_expenses():
    try:
        with open(FILE_NAME, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_expenses(expenses):
    with open(FILE_NAME, "w") as f:
        json.dump(expenses, f, indent=2)


# ---------- input helpers ----------
def get_amount():
    while True:
        try:
            amount = float(input("Amount (Rs): "))
            if amount > 0:
                return amount
            print("Amount must be more than 0.")
        except ValueError:
            print("Please enter a number, e.g. 150 or 49.5")


def confirm_category(model, title):
    """AI suggests a category; the user accepts it or picks another."""
    guess, confidence = suggest_category(model, title)
    print(f"AI suggests: {guess} ({confidence:.0%} sure)")
    for i, cat in enumerate(CATEGORIES, start=1):
        print(f"  {i}. {cat}")
    while True:
        choice = input("Press Enter to accept, or type a category number: ").strip()
        if choice == "":
            return guess
        if choice.isdigit() and 1 <= int(choice) <= len(CATEGORIES):
            return CATEGORIES[int(choice) - 1]
        print("Invalid choice, try again.")


# ---------- features ----------
def add_expense(expenses, model):
    title = input("What did you spend on? ").strip() or "Untitled"
    amount = get_amount()
    category = confirm_category(model, title)
    expenses.append({"date": str(date.today()), "title": title, "amount": amount, "category": category})
    save_expenses(expenses)
    print(f"Added: {title} - Rs {amount:.2f} ({category})")
    return train_categorizer(expenses)  # retrain so the AI learns from this entry


def view_expenses(expenses):
    if not expenses:
        print("No expenses yet.")
        return
    print(f"\n{'No':<4}{'Date':<12}{'Title':<20}{'Category':<12}{'Amount':>10}")
    print("-" * 58)
    for i, e in enumerate(expenses, start=1):
        print(f"{i:<4}{e['date']:<12}{e['title'][:18]:<20}{e['category']:<12}{e['amount']:>10.2f}")
    print("-" * 58)
    print(f"{'Total':<48}{sum(e['amount'] for e in expenses):>10.2f}")


def category_summary(expenses):
    if not expenses:
        print("No expenses yet.")
        return
    totals = {}
    for e in expenses:
        totals[e["category"]] = totals.get(e["category"], 0) + e["amount"]
    grand_total = sum(totals.values())
    print("\nSpending by category:")
    for cat, amt in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<12} Rs {amt:>9.2f}  ({amt / grand_total * 100:.0f}%)")


def show_forecast(expenses):
    result = forecast_next_month(expenses)
    if result is None:
        print("Not enough history yet. Add expenses over a full month (or load demo data).")
        return
    print(f"\nPredicted spending next month: Rs {result['amount']:,.0f}")
    print(f"Based on {result['months_used']} past month(s) using the {result['method']} method.")
    if result["method"] == "trend":
        direction = "rising" if result["change"] > 0 else "falling"
        print(f"Your monthly spending is {direction} by about Rs {abs(result['change']):,.0f} each month.")


def delete_expense(expenses, model):
    view_expenses(expenses)
    if not expenses:
        return model
    choice = input("Enter number to delete (or press Enter to cancel): ")
    if choice.isdigit() and 1 <= int(choice) <= len(expenses):
        removed = expenses.pop(int(choice) - 1)
        save_expenses(expenses)
        print(f"Deleted: {removed['title']}")
        return train_categorizer(expenses)
    print("Cancelled.")
    return model


def load_demo_data(expenses):
    """Add 6 months of sample expenses so the forecast can be tried straight away."""
    if input("Add 6 months of demo data to your list? (y/n): ").strip().lower() != "y":
        print("Cancelled.")
        return False
    rng = random.Random(42)
    sample = [
        ("lunch at canteen", "Food", 90), ("biryani", "Food", 180), ("tea and snacks", "Food", 50),
        ("bus ticket", "Travel", 40), ("metro card recharge", "Travel", 300),
        ("t-shirt", "Shopping", 499), ("wifi bill", "Bills", 600), ("mobile recharge", "Bills", 299),
        ("python book", "Education", 450), ("haircut", "Other", 150),
    ]
    today = date.today()
    for months_back in range(6, 0, -1):
        year, month = today.year, today.month - months_back
        while month <= 0:
            month += 12
            year -= 1
        growth = 1 + (6 - months_back) * 0.05  # spending slowly rises
        for title, category, base in sample:
            for _ in range(rng.randint(1, 3)):
                amount = round(base * growth * rng.uniform(0.8, 1.2))
                expenses.append({"date": date(year, month, rng.randint(1, 28)).isoformat(),
                                 "title": title, "amount": amount, "category": category})
    expenses.sort(key=lambda e: e["date"])
    save_expenses(expenses)
    print("Demo data added. Try option 4 for the forecast!")
    return True


# ---------- main menu ----------
def main():
    expenses = load_expenses()
    model = train_categorizer(expenses)
    menu = """
===== SMART EXPENSE TRACKER =====
1. Add expense (AI suggests category)
2. View all expenses
3. Category summary
4. Forecast next month
5. Delete an expense
6. Load demo data
7. Exit"""
    while True:
        print(menu)
        choice = input("Choose an option (1-7): ").strip()
        if choice == "1":
            model = add_expense(expenses, model)
        elif choice == "2":
            view_expenses(expenses)
        elif choice == "3":
            category_summary(expenses)
        elif choice == "4":
            show_forecast(expenses)
        elif choice == "5":
            model = delete_expense(expenses, model)
        elif choice == "6":
            if load_demo_data(expenses):
                model = train_categorizer(expenses)
        elif choice == "7":
            print("Bye! Spend smart.")
            break
        else:
            print("Please choose 1 to 7.")


if __name__ == "__main__":
    main()
