from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Define Q&A data
data = [
    # Introduction
    {"question": "Hi! Am I speaking with Mr./Ms. _______?", 
     "answer": "Yes, this is Ketto. How are you today?"},
    {"question": "How are you doing today?", 
     "answer": "Nice to hear that! I’m fine, thank you for asking."},
    {"question": "Can you call me later?", 
     "answer": "I completely understand you must be busy right now, but I only need 2 minutes of your time."},

    # Purpose of the Call
    {"question": "Why are you calling today?", 
     "answer": "We’d like to thank you for your previous donation. It has made a significant impact on someone’s life. We’re here to talk about an urgent campaign."},
    {"question": "What is the purpose of this call?", 
     "answer": "We are running a Medical and Education SIP campaign to save underprivileged children’s lives through monthly contributions."},

    # Donation Details
    {"question": "How much do you need me to donate?", 
     "answer": "We request a small monthly donation of Rs. 200–300 to support our cause."},
    {"question": "Can I donate more than the suggested amount?", 
     "answer": "Of course! Your generosity will help us reach more people in need."},
    {"question": "Can I make a one-time donation instead of a recurring one?", 
     "answer": "Yes, you can make a one-time donation, but recurring donations create a long-lasting impact."},
    {"question": "Can I donate anonymously?", 
     "answer": "Yes, your donation can remain anonymous if you wish."},

    # Tax Benefits
    {"question": "What are the tax benefits of donating?", 
     "answer": "Under the 80G tax benefit scheme, your donations are deductible from your taxable income."},
    {"question": "How does the 80G tax rebate work?", 
     "answer": "The 80G tax rebate allows you to reduce your taxable income based on your donation amount. This is valid for contributions to approved charities."},
    {"question": "Will I get a receipt for the donation?", 
     "answer": "Yes, you’ll receive an acknowledgment and a receipt for tax filing purposes."},

    # Donation Process
    {"question": "How do I proceed with the donation?", 
     "answer": "We’ll send a UPI payment request to your registered number. It takes less than a minute to complete."},
    {"question": "Can I pay via credit card?", 
     "answer": "Yes, you can donate via credit card, debit card, net banking, or UPI."},
    {"question": "Can I use my company’s name for the donation?", 
     "answer": "Yes, corporate donations are welcome and eligible for 80G tax benefits."},
    {"question": "How can I confirm my donation went through?", 
     "answer": "You will receive a confirmation message and email once your donation is successful."},

    # Follow-Up After Donation
    {"question": "What happens after I donate?", 
     "answer": "You’ll receive updates about the impact of your donation and how it is helping beneficiaries."},
    {"question": "Will I get updates about my donations?", 
     "answer": "Yes, we regularly send updates via email or WhatsApp on how your contributions are being utilized."},
    {"question": "Can I get a summary of my contributions?", 
     "answer": "Absolutely! We can send you an annual summary of your donations."},

    # Campaign Impact
    {"question": "What is the impact of this campaign?", 
     "answer": "Your donations help provide medical care, education, and basic necessities to underprivileged children."},
    {"question": "Can you share some success stories?", 
     "answer": "Yes, many children have received life-saving treatments and education through our campaigns. We can share detailed stories upon request."},
    {"question": "What is the current status of this campaign?", 
     "answer": "We are urgently raising funds to support ongoing treatments and education for children in need."},

    # Additional Causes
    {"question": "What are the other causes I can support?", 
     "answer": "We have campaigns for education, food and hunger relief, animal welfare, elderly care, and women empowerment."},
    {"question": "How can I support education campaigns?", 
     "answer": "You can fund the education of underprivileged children up to the 12th standard."},
    {"question": "How can I support food and hunger campaigns?", 
     "answer": "Your contributions can provide mid-day meals to slum children, ensuring proper nutrition."},
    {"question": "How can I support animal welfare?", 
     "answer": "You can support shelter, food, and medical aid for street animals."},
    {"question": "How can I support elderly care?", 
     "answer": "You can provide basic needs like food, clothing, and shelter to senior citizens in need."},
    {"question": "How can I support women empowerment?", 
     "answer": "You can help women learn skills like sewing, basic computer skills, and beautician training to become financially independent."},

    # Closing and Feedback
    {"question": "Why should I contribute today?", 
     "answer": "This is an urgent matter. Your timely contribution can save lives."},
    {"question": "Can I donate at a later date?", 
     "answer": "While we’d appreciate your immediate support, we can schedule a callback at your convenience."},
    {"question": "How can I provide feedback about the donation process?", 
     "answer": "You can share your feedback directly with us to help improve the process."},
    {"question": "Can I recommend someone else to donate?", 
     "answer": "Yes, referrals are welcome, and we can provide details to them as well."},

    # General Queries
    {"question": "How is my donation used?", 
     "answer": "Your donation is used to fund treatments, education, and basic needs for underprivileged individuals."},
    {"question": "Are your campaigns audited?", 
     "answer": "Yes, all our campaigns are audited to ensure transparency."},
    {"question": "Can I visit your office or beneficiaries?", 
     "answer": "Yes, you are welcome to visit our office or see the beneficiaries your donations support."},
    {"question": "Can I stop recurring donations?", 
     "answer": "Yes, you can stop anytime by notifying us, though we encourage continued support for long-term impact."}
]

# This is 40 questions to showcase the format. Let me know if you'd like the dataset expanded to 100 Q&A pairs!


# Extract questions and answers
questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Use SentenceTransformers for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast
question_embeddings = embedding_model.encode(questions)

# Save embeddings and corresponding data
embeddings_file = "data/raw/question_embeddings.pkl"
data_file = "data/raw/qa_data.pkl"

# Save embeddings to a file
with open(embeddings_file, "wb") as f:
    pickle.dump(question_embeddings, f)

# Save the questions and answers for reference
with open(data_file, "wb") as f:
    pickle.dump(data, f)

print(f"Embeddings saved to {embeddings_file}.")
print(f"Q&A data saved to {data_file}.")
