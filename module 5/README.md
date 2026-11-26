# Smart Expense Tracker (Module 05 - AI Tools & Mini Project)

An upgrade of my Module 02 Expense Tracker. It now has two AI features built with scikit-learn:

1. **Auto-categorisation:** type what you spent on, and the AI suggests the category (you can accept or change it).
2. **Spending forecast:** the AI predicts next month's total spending from your past months.

**Author:** Barkavi R

## What changed from the basic tracker

| Feature | Module 02 (basic) | Module 05 (smart) |
|---|---|---|
| Choosing a category | You pick it every time | AI suggests it with a confidence % |
| Learning | None | Retrains on your own entries |
| Future spending | Not available | Forecast for next month |
| Code structure | One file | UI in `smart_expense_tracker.py`, AI in `expense_ai.py` |
| Testing | Manual | `test_expense_ai.py` |

## How the AI works (plain language)

| Feature | Analogy | Technique |
|---|---|---|
| Auto-categoriser | Like a shopkeeper who has seen "biryani", "tea", "dosa" so many times that he knows a new Food item when he sees one | Text is turned into numbers with **TF-IDF** (character n-grams), then a **Logistic Regression** classifier predicts the category |
| Learning over time | The shopkeeper keeps learning from every bill you confirm | The model retrains with your saved expenses after each change |
| Forecast | Drawing the best straight line through your past monthly totals and extending it one month | **Linear Regression** on monthly totals (uses a simple average if there are fewer than 3 months) |

## How well does it work?

Tested on 24 expense titles the model had never seen:

- **Accuracy: 22 / 24 (92%)**
- Misses: "taxi" (guessed Food) and "broadband" (guessed Food). Very short or unusual words are hard because the model only has about 100 starter examples.
- That is why the app always shows a confidence % and lets you correct it. Every correction makes it smarter.

**Limitations:** the forecast is a simple trend, so it cannot know about festivals, fee payments, or one-time big purchases.

## How to run

1. Install Python 3 from [python.org](https://www.python.org/downloads/).
2. Install the libraries:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python smart_expense_tracker.py
```

4. Try option **6 (Load demo data)** and then option **4 (Forecast)** to see the AI in action.
5. Run the checks: `python test_expense_ai.py`

## Sample session

```
Choose an option (1-7): 1
What did you spend on? biriyani dinner
Amount (Rs): 250
AI suggests: Food (84% sure)
Press Enter to accept, or type a category number:
Added: biriyani dinner - Rs 250.00 (Food)

Choose an option (1-7): 4
Predicted spending next month: Rs 7,784
Based on 6 past month(s) using the trend method.
Your monthly spending is rising by about Rs 343 each month.
```

## Project files

| File | Purpose |
|---|---|
| `smart_expense_tracker.py` | Menu, saving/loading data, user interaction |
| `expense_ai.py` | AI code: category classifier and forecast |
| `test_expense_ai.py` | Accuracy and forecast checks |
| `requirements.txt` | Libraries needed |

## Exploring AI tools

I explored AI assistants (ChatGPT, Gemini, Microsoft Copilot, Claude) to see how they help with coding, research, and productivity.

| Area | How AI tools helped | What I learned |
|---|---|---|
| Coding | Explained scikit-learn code, suggested features, helped debug errors | Always run and test the code yourself, because AI can be confidently wrong |
| Research | Summarised ideas like TF-IDF and linear regression in simple words | Check important facts against official docs |
| Productivity | Planned the project steps and drafted this README | Clear, specific prompts give much better answers |

I reviewed, ran, and tested all the code in this repo myself.

## Future ideas

- Use an LLM API to understand sentences like "paid Rs 200 for auto to college"
- Monthly budget alerts based on the forecast
- Charts with Matplotlib and a simple web UI with Streamlit
