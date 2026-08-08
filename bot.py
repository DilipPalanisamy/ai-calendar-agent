import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from parser import parse_schedule_message
from calendar_service import (
    create_google_calendar_event,
    check_calendar_conflict,
    find_event_by_title,
    delete_google_calendar_event,
    reschedule_google_calendar_event,
    get_primary_calendar_timezone,
    get_calendar_service,
)

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I'm your AI Calendar Assistant.\n\n"
        "I can help you with:\n"
        "👉 **Creating:** 'Team sync tomorrow at 3pm'\n"
        "👉 **Rescheduling:** 'Move team sync tomorrow to 5pm'\n"
        "👉 **Deleting:** 'Cancel team sync tomorrow'\n"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.reply_text("⏳ Processing your request...")

    try:
        parsed_data = parse_schedule_message(user_text)
        reply_lines = []
        calendar_timezone = get_primary_calendar_timezone(get_calendar_service())

        for event in parsed_data.events:
            action = getattr(event, 'action', 'CREATE').upper()

            # --- DELETE ACTION ---
            if action == "DELETE":
                target_event = find_event_by_title(event.event_name)
                if target_event and delete_google_calendar_event(target_event['id']):
                    reply_lines.append(f"🗑️ **Deleted Event:** '{target_event.get('summary')}'")
                else:
                    reply_lines.append(f"❌ Could not find event matching '{event.event_name}' to delete.")

            # --- RESCHEDULE ACTION ---
            elif action == "RESCHEDULE":
                target_event = find_event_by_title(event.event_name)
                if target_event:
                    link = reschedule_google_calendar_event(target_event['id'], event.start_time, event.end_time)
                    reply_lines.append(
                        f"🔄 **Rescheduled Event:** '{target_event.get('summary')}'\n"
                        f"⏰ New Time ({calendar_timezone}): {event.start_time} to {event.end_time}\n"
                        f"🔗 [View in Google Calendar]({link})\n"
                    )
                else:
                    reply_lines.append(f"❌ Could not find event matching '{event.event_name}' to reschedule.")

            # --- CREATE ACTION ---
            else:
                conflicts = check_calendar_conflict(event.start_time, event.end_time)
                if conflicts:
                    conflict_names = ", ".join([f"'{c}'" for c in conflicts])
                    reply_lines.append(f"⚠️ **Conflict Detected:** Overlaps with {conflict_names}.\n")

                link = create_google_calendar_event(event)
                reply_lines.append(
                    f"✅ **Created Event:** {event.event_name}\n"
                    f"⏰ {event.start_time} to {event.end_time} ({calendar_timezone})\n"
                    f"🔗 [View in Google Calendar]({link})\n"
                )

        await update.message.reply_markdown('\n'.join(reply_lines))

    except Exception as e:
        await update.message.reply_text(f"❌ Failed to process request: {str(e)}")


if __name__ == '__main__':
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not found in .env file!")
        exit(1)

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("🤖 Telegram Bot is running! Press Ctrl+C to stop.")
    app.run_polling()