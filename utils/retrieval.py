from sklearn.metrics.pairwise import cosine_similarity

def retrieve_context(query, question_embeddings, data, embedding_model, top_k=1):
    """
    Retrieve the most relevant context for a query.
    :param query: User's query string
    :param question_embeddings: Precomputed embeddings of the questions
    :param data: Q&A data
    :param embedding_model: Sentence embedding model
    :param top_k: Number of top contexts to retrieve
    :return: Most relevant context
    """
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]

    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, question_embeddings)[0]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]

    results = [{"question": questions[i], "answer": answers[i], "similarity": similarities[i]} for i in top_indices]
    return results[0]["answer"] if results else "No relevant context found."
