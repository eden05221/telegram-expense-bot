import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from openai import OpenAI
from dotenv import load_dotenv


# =========================================================
# טעינת משתני סביבה
# =========================================================
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# אפשר לשנות למודל אחר אם תרצי
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

DATA_FILE = "expenses_data.json"

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment variables.")

# OpenAI הוא אופציונלי - אם אין מפתח, עדיין אפשר לעבוד עם פקודות ידניות
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================================================
# לוגים
# =========================================================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =========================================================
# ניהול קובץ נתונים
# =========================================================
def load_data() -> Dict[str, Any]:
    """טוען את הנתונים מקובץ JSON. אם הקובץ לא קיים, מחזיר מבנה ריק."""
    if not os.path.exists(DATA_FILE):
        return {"chats": {}}

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        return {"chats": {}}


def save_data(data: Dict[str, Any]) -> None:
    """שומר את כל הנתונים לקובץ JSON."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_chat_expenses(chat_id: int) -> List[Dict[str, Any]]:
    """מחזיר את רשימת ההוצאות של צ'אט מסוים."""
    data = load_data()
    chat_key = str(chat_id)

    if chat_key not in data["chats"]:
        data["chats"][chat_key] = {"expenses": []}
        save_data(data)

    return data["chats"][chat_key]["expenses"]


def set_chat_expenses(chat_id: int, expenses: List[Dict[str, Any]]) -> None:
    """מעדכן את רשימת ההוצאות של צ'אט מסוים."""
    data = load_data()
    chat_key = str(chat_id)

    if chat_key not in data["chats"]:
        data["chats"][chat_key] = {"expenses": []}

    data["chats"][chat_key]["expenses"] = expenses
    save_data(data)


def add_expense(
    chat_id: int,
    amount: float,
    currency: str,
    category: str,
    description: str,
    country: str = "",
    city: str = "",
) -> Dict[str, Any]:
    """מוסיף הוצאה חדשה ושומר אותה."""
    expenses = get_chat_expenses(chat_id)

    expense = {
        "id": len(expenses) + 1,
        "amount": amount,
        "currency": currency.upper(),
        "category": category.lower(),
        "description": description,
        "country": country,
        "city": city,
        "created_at": datetime.now().isoformat(),
    }

    expenses.append(expense)
    set_chat_expenses(chat_id, expenses)
    return expense


def delete_last_expense(chat_id: int) -> Optional[Dict[str, Any]]:
    """מוחק את ההוצאה האחרונה."""
    expenses = get_chat_expenses(chat_id)
    if not expenses:
        return None

    removed = expenses.pop()
    set_chat_expenses(chat_id, expenses)
    return removed


# =========================================================
# פונקציות עזר לחישובים
# =========================================================
def summarize_expenses(expenses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    מסכם את ההוצאות.
    כרגע הסיכום מתבצע לפי מטבע בנפרד.
    """
    totals_by_currency = {}
    totals_by_category = {}
    totals_by_country = {}

    for exp in expenses:
        currency = exp.get("currency", "EUR").upper()
        category = exp.get("category", "other").lower()
        country = exp.get("country", "").strip() or "unknown"

        amount = float(exp.get("amount", 0))

        totals_by_currency[currency] = totals_by_currency.get(currency, 0) + amount

        if category not in totals_by_category:
            totals_by_category[category] = {}
        totals_by_category[category][currency] = (
            totals_by_category[category].get(currency, 0) + amount
        )

        if country not in totals_by_country:
            totals_by_country[country] = {}
        totals_by_country[country][currency] = (
            totals_by_country[country].get(currency, 0) + amount
        )

    return {
        "count": len(expenses),
        "totals_by_currency": totals_by_currency,
        "totals_by_category": totals_by_category,
        "totals_by_country": totals_by_country,
    }


def format_totals_dict(totals: Dict[str, float]) -> str:
    """ממיר dict של סכומים לפי מטבע לטקסט יפה."""
    if not totals:
        return "0"
    return ", ".join([f"{amount:.2f} {currency}" for currency, amount in totals.items()])


def format_expense(expense: Dict[str, Any]) -> str:
    """מייצר טקסט ידידותי עבור הוצאה אחת."""
    country_part = f" | {expense['country']}" if expense.get("country") else ""
    city_part = f", {expense['city']}" if expense.get("city") else ""
    return (
        f"#{expense['id']} - {expense['amount']:.2f} {expense['currency']} | "
        f"{expense['category']} | {expense['description']}{country_part}{city_part}"
    )


# =========================================================
# OpenAI - הבנת הודעות חופשיות
# =========================================================
def parse_user_message_with_ai(user_text: str) -> Optional[Dict[str, Any]]:
    """
    משתמש ב-OpenAI כדי להבין האם ההודעה היא:
    1. add_expense
    2. ask_question
    3. unknown
    """
    if not openai_client:
        return None

    system_prompt = """
You are an expense-tracking parser for a Telegram bot.

Your job:
Return ONLY valid JSON with this schema:

{
  "intent": "add_expense" | "ask_question" | "unknown",
  "amount": number | null,
  "currency": string | null,
  "category": string | null,
  "description": string | null,
  "country": string | null,
  "city": string | null,
  "question": string | null
}

Rules:
- If the user describes a spending event, use intent "add_expense".
- If the user asks about totals, trends, categories, countries, or previous spending, use intent "ask_question".
- If unclear, use "unknown".
- Categories should be one of:
  food, accommodation, transport, clothes, activities, shopping, misc
- If category is uncertain, use "misc".
- If currency is not written, infer if reasonable, otherwise use "EUR".
- Keep descriptions short.
- Return JSON only. No markdown.
"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()
        return json.loads(content)

    except Exception as e:
        logger.error(f"AI parsing error: {e}")
        return None


def answer_question_with_ai(question: str, expenses: List[Dict[str, Any]]) -> str:
    """
    משתמש ב-AI כדי לענות על שאלות חופשיות על בסיס רשימת ההוצאות בלבד.
    """
    if not openai_client:
        return (
            "OpenAI is not configured. "
            "You can still use /total, /category, /list manually."
        )

    compact_expenses = [
        {
            "amount": exp["amount"],
            "currency": exp["currency"],
            "category": exp["category"],
            "description": exp["description"],
            "country": exp.get("country", ""),
            "city": exp.get("city", ""),
            "created_at": exp["created_at"],
        }
        for exp in expenses
    ]

    system_prompt = """
You are an assistant for an expense tracking Telegram bot.
Answer ONLY based on the provided expenses data.
Do not invent expenses that are not present.
If the answer requires grouping by currency, keep currencies separate.
Be concise but helpful.
"""

    user_prompt = f"""
Question:
{question}

Expenses data:
{json.dumps(compact_expenses, ensure_ascii=False, indent=2)}
"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"AI answering error: {e}")
        return "Sorry, I failed to answer that question right now."


# =========================================================
# פקודות טלגרם
# =========================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Hi! I am your travel expense tracker bot ✈️💸\n\n"
        "You can add expenses with a command like:\n"
        "/add 18 EUR food pizza Italy Rome\n\n"
        "Format:\n"
        "/add <amount> <currency> <category> <description> [country] [city]\n\n"
        "Examples:\n"
        "/add 12 EUR food coffee Italy Rome\n"
        "/add 95 EUR accommodation hotel Italy Florence\n\n"
        "Useful commands:\n"
        "/list\n"
        "/total\n"
        "/category food\n"
        "/delete_last\n"
        "/ask How much did we spend in Italy?\n\n"
        "You can also send free text like:\n"
        "'Spent 20 euro on pasta in Rome'\n"
        "'How much did we spend on food so far?'"
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Commands:\n\n"
        "/start - start the bot\n"
        "/help - show help\n"
        "/add <amount> <currency> <category> <description> [country] [city]\n"
        "/list - show all expenses\n"
        "/total - show total spending\n"
        "/category <name> - show totals for one category\n"
        "/delete_last - remove last expense\n"
        "/ask <question> - ask a free-text question about expenses\n\n"
        "Allowed categories:\n"
        "food, accommodation, transport, clothes, activities, shopping, misc"
    )
    await update.message.reply_text(text)


async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    דוגמה:
    /add 18 EUR food pizza Italy Rome
    """
    args = context.args

    if len(args) < 4:
        await update.message.reply_text(
            "Usage:\n"
            "/add <amount> <currency> <category> <description> [country] [city]\n\n"
            "Example:\n"
            "/add 18 EUR food pizza Italy Rome"
        )
        return

    try:
        amount = float(args[0])
        currency = args[1].upper()
        category = args[2].lower()
        description = args[3]

        country = args[4] if len(args) >= 5 else ""
        city = args[5] if len(args) >= 6 else ""

        expense = add_expense(
            chat_id=update.effective_chat.id,
            amount=amount,
            currency=currency,
            category=category,
            description=description,
            country=country,
            city=city,
        )

        await update.message.reply_text(
            "Expense added successfully ✅\n"
            f"{format_expense(expense)}"
        )

    except ValueError:
        await update.message.reply_text("Amount must be a valid number.")


async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    expenses = get_chat_expenses(update.effective_chat.id)

    if not expenses:
        await update.message.reply_text("No expenses yet.")
        return

    lines = ["Your expenses:\n"]
    for exp in expenses[-20:]:  # מציגים עד 20 אחרונות
        lines.append(format_expense(exp))

    await update.message.reply_text("\n".join(lines))


async def total_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    expenses = get_chat_expenses(update.effective_chat.id)
    summary = summarize_expenses(expenses)

    if summary["count"] == 0:
        await update.message.reply_text("No expenses yet.")
        return

    totals_text = format_totals_dict(summary["totals_by_currency"])

    lines = [
        f"Total expenses: {summary['count']}",
        f"Total spent: {totals_text}",
        "",
        "By category:",
    ]

    for category, totals in summary["totals_by_category"].items():
        lines.append(f"- {category}: {format_totals_dict(totals)}")

    await update.message.reply_text("\n".join(lines))


async def category_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /category <name>")
        return

    requested_category = context.args[0].lower()
    expenses = get_chat_expenses(update.effective_chat.id)

    filtered = [e for e in expenses if e.get("category", "").lower() == requested_category]

    if not filtered:
        await update.message.reply_text(f"No expenses found for category '{requested_category}'.")
        return

    totals = {}
    for exp in filtered:
        currency = exp["currency"]
        totals[currency] = totals.get(currency, 0) + float(exp["amount"])

    lines = [
        f"Category: {requested_category}",
        f"Total: {format_totals_dict(totals)}",
        "",
        "Items:",
    ]
    for exp in filtered[-20:]:
        lines.append(format_expense(exp))

    await update.message.reply_text("\n".join(lines))


async def delete_last_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    removed = delete_last_expense(update.effective_chat.id)

    if not removed:
        await update.message.reply_text("There is no expense to delete.")
        return

    await update.message.reply_text(
        "Deleted last expense 🗑️\n"
        f"{format_expense(removed)}"
    )


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args).strip()

    if not question:
        await update.message.reply_text("Usage: /ask <your question>")
        return

    expenses = get_chat_expenses(update.effective_chat.id)

    if not expenses:
        await update.message.reply_text("No expenses yet.")
        return

    answer = answer_question_with_ai(question, expenses)
    await update.message.reply_text(answer)


# =========================================================
# טיפול בהודעות חופשיות
# =========================================================
async def handle_free_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    אם המשתמש לא שלח פקודה, ננסה להבין עם AI:
    - האם זו הוצאה
    - האם זו שאלה
    """
    user_text = update.message.text.strip()

    # אם אין OpenAI, נסביר למשתמש להשתמש בפקודות
    if not openai_client:
        await update.message.reply_text(
            "I did not understand that automatically.\n"
            "Please use a command like /add or /total.\n"
            "If you want AI parsing, configure OPENAI_API_KEY."
        )
        return

    parsed = parse_user_message_with_ai(user_text)

    if not parsed:
        await update.message.reply_text(
            "Sorry, I could not understand that.\n"
            "Try /add 18 EUR food pizza Italy Rome"
        )
        return

    intent = parsed.get("intent")

    if intent == "add_expense":
        amount = parsed.get("amount")
        currency = parsed.get("currency") or "EUR"
        category = parsed.get("category") or "misc"
        description = parsed.get("description") or "expense"
        country = parsed.get("country") or ""
        city = parsed.get("city") or ""

        if amount is None:
            await update.message.reply_text(
                "I think you tried to add an expense, but I could not find the amount."
            )
            return

        expense = add_expense(
            chat_id=update.effective_chat.id,
            amount=float(amount),
            currency=currency,
            category=category,
            description=description,
            country=country,
            city=city,
        )

        await update.message.reply_text(
            "Got it, I added this expense ✅\n"
            f"{format_expense(expense)}"
        )
        return

    if intent == "ask_question":
        question = parsed.get("question") or user_text
        expenses = get_chat_expenses(update.effective_chat.id)

        if not expenses:
            await update.message.reply_text("No expenses yet.")
            return

        answer = answer_question_with_ai(question, expenses)
        await update.message.reply_text(answer)
        return

    await update.message.reply_text(
        "I am not sure what you meant.\n"
        "Try one of these:\n"
        "/add 18 EUR food pizza Italy Rome\n"
        "/total\n"
        "/ask How much did we spend on food?"
    )


# =========================================================
# Error handler
# =========================================================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)


# =========================================================
# Main
# =========================================================
def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # פקודות
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("add", add_command))
    app.add_handler(CommandHandler("list", list_command))
    app.add_handler(CommandHandler("total", total_command))
    app.add_handler(CommandHandler("category", category_command))
    app.add_handler(CommandHandler("delete_last", delete_last_command))
    app.add_handler(CommandHandler("ask", ask_command))

    # הודעות טקסט רגילות
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_free_text))

    # טיפול בשגיאות
    app.add_error_handler(error_handler)

    logger.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()