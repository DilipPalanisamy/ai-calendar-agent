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
    find_gmail_drive_or_internship_messages,
)

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

pending_gmail_events = {}


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
        chat_id = update.effective_chat.id
        normalized_text = user_text.strip().lower()
        if normalized_text in {"approve", "approved", "yes", "add it", "add to calendar"} or normalized_text.startswith("approve "):
            pending_events = pending_gmail_events.pop(chat_id, None)
            if not pending_events:
                messages = find_gmail_drive_or_internship_messages()
                pending_events = []
                for message in messages:
                    email_text = f"{message['subject']}\n{message['snippet']}"
                    pending_events.extend(parse_schedule_message(email_text, "Asia/Kolkata").events)
            if not pending_events:
                await update.message.reply_text("There is no unread Gmail event waiting for approval.")
                return

            links = [create_google_calendar_event(event) for event in pending_events]
            await update.message.reply_text(
                "✅ Added the approved Gmail event(s) to Google Calendar.\n"
                + "\n".join(link for link in links if link)
            )
            return

        if any(word in normalized_text for word in ["gmail", "email", "drive", "internship"]):
            messages = find_gmail_drive_or_internship_messages()
            if not messages:
                await update.message.reply_text("No unread Drive or internship messages found in Gmail.")
                return

            pending_gmail_events[chat_id] = []
            lines = ["📧 Gmail messages found (Drive/internship):"]
            for message in messages:
                email_text = f"{message['subject']}\n{message['snippet']}"
                parsed_email = parse_schedule_message(email_text, "Asia/Kolkata")
                pending_gmail_events[chat_id].extend(parsed_email.events)
                lines.append(
                    f"\nSubject: {message['subject']}\n"
                    f"From: {message['from']}\n"
                    f"Message: {message['display_snippet']}"
                )
            lines.append("\nReply 'approve' to add these detected event(s) to your calendar.")
            await update.message.reply_text("\n".join(lines))
            return

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