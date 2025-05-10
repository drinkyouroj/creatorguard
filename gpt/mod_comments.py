import os
import json
import sqlite3
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-turbo')

def classify_comment(comment_text):
    """
    Use GPT to classify a comment's sentiment and suggest moderation action.
    
    Args:
        comment_text (str): The text of the comment to classify
    
    Returns:
        dict: A dictionary with classification details
    """
    prompt = f"""Classify the following YouTube comment and provide a detailed analysis:

Comment: "{comment_text}"

Please provide a response with the following JSON structure:
{{
    "classification": "supportive|constructive|neutral|critical|toxic",
    "mod_action": "respond|flag|hide|ignore",
    "reason": "Explanation of the classification and recommended action",
    "suggested_reply": "Optional suggested response (if applicable)"
}}

Your analysis should be nuanced, considering context and tone."""

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful AI trained to moderate YouTube comments with empathy and nuance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent results
        )
        
        # Parse the GPT response
        gpt_result = json.loads(response.choices[0].message.content)
        return gpt_result
    
    except Exception as e:
        print(f"Error classifying comment: {e}")
        return {
            "classification": "error",
            "mod_action": "manual_review",
            "reason": str(e),
            "suggested_reply": None
        }

def process_comments_json(json_file='comments/comments.json', db_path='creatorguard.db'):
    """
    Process comments from a JSON file and update the SQLite database with GPT classifications.
    
    Args:
        json_file (str): Path to the JSON file containing comments
        db_path (str): Path to the SQLite database
    """
    # Read comments from JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        comments = json.load(f)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Process each comment
    for comment in comments:
        # Classify comment with GPT
        classification = classify_comment(comment['text'])
        
        # Update database with classification
        try:
            cursor.execute("""
                UPDATE comments 
                SET classification = ?, 
                    mod_action = ?, 
                    reason = ?, 
                    suggested_reply = ?
                WHERE id = ?
            """, (
                classification['classification'],
                classification['mod_action'],
                classification['reason'],
                classification['suggested_reply'],
                comment['id']
            ))
        except sqlite3.Error as e:
            print(f"Database update error for comment {comment['id']}: {e}")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"✅ Processed {len(comments)} comments with GPT moderation.")

def main():
    process_comments_json()

if __name__ == '__main__':
    main()
