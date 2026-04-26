def get_chat_history(chat_history):
    """
    Convert chat history list into text
    """
    history_text = ""

    for user, bot in chat_history:
        history_text += f"User: {user}\n"
        history_text += f"Assistant: {bot}\n"

    return history_text