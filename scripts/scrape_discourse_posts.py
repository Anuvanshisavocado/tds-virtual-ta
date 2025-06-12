import os
import json
import datetime
from dotenv import load_dotenv

# Load environment variables (e.g., for DISCOURSE_API_KEY if you were to use it)
load_dotenv()

# Base URL for IITM Discourse (used for constructing dummy URLs)
DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"

def get_discourse_posts_dummy(start_date_str, end_date_str):
    """
    Simulates fetching Discourse posts within a date range and saves them
    as text files, along with a metadata JSON file.
    """
    output_dir = "data/discourse_posts"
    os.makedirs(output_dir, exist_ok=True)

    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()

    all_posts_metadata = []

    print(f"Simulating Discourse posts scraping from {start_date} to {end_date}...")

    # DUMMY DATA: This simulates the actual posts you'd get from Discourse API
    # The 'raw_content' is what a student or TA would post.
    # The 'url' is critical for the final API response.
    dummy_posts = [
        {
            "topic_id": 155001, "post_id": 15500101, "topic_title": "GA1 Question 5 Clarification",
            "raw_content": "I'm stuck on GA1 Q5 regarding Pandas DataFrame manipulation. How do I filter rows based on multiple conditions efficiently?",
            "created_at": "2025-01-15", "username": "student_alpha",
            "url": f"{DISCOURSE_BASE_URL}/t/ga1-question-5-clarification/155001/1"
        },
        {
            "topic_id": 155001, "post_id": 15500102, "topic_title": "GA1 Question 5 Clarification",
            "raw_content": "You can use boolean indexing with the `&` operator for multiple conditions. Example: `df[(df['col1'] > 10) & (df['col2'] == 'value')]`. Make sure to wrap each condition in parentheses.",
            "created_at": "2025-01-16", "username": "ta_beta",
            "url": f"{DISCOURSE_BASE_URL}/t/ga1-question-5-clarification/155001/2"
        },
        {
            "topic_id": 155939, "post_id": 15593901, "topic_title": "GA5 Question 8 Clarification (OpenAI Model)",
            "raw_content": "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo for GA5 Question 8? I'm confused about which model is allowed.",
            "created_at": "2025-04-10", "username": "student_charlie",
            "url": f"{DISCOURSE_BASE_URL}/t/ga5-question-8-clarification/155939/1"
        },
        {
            "topic_id": 155939, "post_id": 15593902, "topic_title": "GA5 Question 8 Clarification (OpenAI Model)",
            "raw_content": "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question. My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate.",
            "created_at": "2025-04-11", "username": "ta_delta",
            "url": f"{DISCOURSE_BASE_URL}/t/ga5-question-8-clarification/155939/3"
        },
        {
            "topic_id": 155939, "post_id": 15593903, "topic_title": "GA5 Question 8 Clarification (OpenAI Model)",
            "raw_content": "Just to confirm, we should use the model that’s mentioned in the question, right? So, `gpt-3.5-turbo-0125` is the official model.",
            "created_at": "2025-04-12", "username": "student_echo",
            "url": f"{DISCOURSE_BASE_URL}/t/ga5-question-8-clarification/155939/4"
        }
    ]

    # Filter dummy posts by date and save them
    for post_data in dummy_posts:
        post_date = datetime.datetime.strptime(post_data["created_at"], "%Y-%m-%d").date()
        if start_date <= post_date <= end_date:
            # Save the raw content of the post to a text file
            file_path = os.path.join(output_dir, f"discourse_post_{post_data['post_id']}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(post_data["raw_content"])
            print(f"Saved Discourse post content to {file_path}")

            # Store the metadata (including URL)
            all_posts_metadata.append(post_data)

    # Save all posts metadata to a single JSON file
    metadata_file_path = os.path.join(output_dir, "all_discourse_posts_metadata.json")
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump(all_posts_metadata, f, indent=4)
    print(f"\nSaved all Discourse posts metadata to {metadata_file_path}")

    return all_posts_metadata

if __name__ == "__main__":
    # Define the date range as specified in the problem description
    start_date = "2025-01-01"
    end_date = "2025-04-14"
    fetched_posts = get_discourse_posts_dummy(start_date, end_date)
    print(f"\nTotal dummy Discourse posts included in metadata: {len(fetched_posts)}")