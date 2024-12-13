def create_prompt(context, question, user_name=None):
    """
    Create a concise prompt for Falcon model with a conversational flow.
    :param context: Retrieved context
    :param question: User's query
    :param user_name: Name of the user (optional, for personalizing responses)
    :return: Prompt string
    """
    if user_name:
        return (
            f"You are Ketto, a helpful assistant for donation and social campaign inquiries. "
            f"Respond politely, directly, and concisely to the user's question. "
            f"Do not repeat your response or the context unnecessarily.\n\n"
            f"Context: {context}\n\n"
            f"User: {user_name}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    else:
        return (
            f"You are Ketto, a helpful assistant for donation and social campaign inquiries. "
            f"Start by introducing yourself and asking for the user's name. Once you know their name, "
            f"respond politely, directly, and concisely to their questions. Avoid repeating responses.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
