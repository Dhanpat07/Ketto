def generate_detailed_response(context, question, tokenizer, model, prompt_creator):
    """
    Generate a focused response using Falcon.
    :param context: Retrieved context
    :param question: User's query
    :param tokenizer: Tokenizer for Falcon
    :param model: Falcon model
    :param prompt_creator: Function to create prompts
    :return: Generated response
    """
    # Create the prompt using the specified function
    prompt = prompt_creator(context, question)

    # Generate response
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_length=200,  # Limit the response length
        temperature=0.7,  # Add slight randomness for natural responses
        top_p=0.9,  # Focus on high-probability tokens
        do_sample=True,  # Enable sampling for varied responses
        num_return_sequences=1  # Generate a single response
    )

    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the relevant part of the response
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    # Remove redundancy if the response repeats itself
    response_lines = response.split("\n")
    cleaned_response = response_lines[0].strip() if len(response_lines) > 1 else response.strip()

    return cleaned_response
