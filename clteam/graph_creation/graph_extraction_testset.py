import os
import pandas as pd
import openai
import json
import time
from tqdm import tqdm
from typing import List, Dict, Tuple
import tiktoken
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to load data
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

# Extract texts regardless of language
def extract_texts(conversation: pd.DataFrame) -> pd.DataFrame:
    texts = []
    senders = []
    for _, row in conversation.iterrows():
        texts.append(row['source'])
        senders.append(row['sender'])
    return pd.DataFrame({'text': texts, 'sender': senders})

# Set OpenAI API key
openai.api_key = 'API KEY HERE'

# Rate limiting settings
RATE_LIMIT = 5000  # requests per minute
TOKEN_LIMIT = 2000000  # tokens per minute
REQUEST_INTERVAL = 60 / RATE_LIMIT

# Use tiktoken for counting tokens
def num_tokens_from_messages(messages, model="gpt-4o-mini"):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = 0
    for message in messages:
        tokens += len(encoding.encode(message['content']))
    return tokens

# Log errors
def log_error(conversation_id: str, error_message: str, error_log_path: str):
    with open(error_log_path, 'a') as log_file:
        log_file.write(f"{conversation_id}: {error_message}\n")

# Retry decorator
def retry_on_exception(retries=3, delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        print(f"Retrying due to: {e}. Attempt {attempt + 1}/{retries}")
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator

@retry_on_exception(retries=5, delay=10)
def extract_entities_relationships_and_context(conversation_id: str, conversation_text: str, total_tokens: int, start_time: float, error_log_path: str) -> Tuple[Dict, int]:
    try:
        system_prompt = """
        You will translate and analyze a dialogue and break it down into triples consisting of a subject, predicate, and object. Each triple should capture the essence of interactions between speakers. 

        Translation: If the sentence is not in English, translate it and ensure the triple content is translated too. All output should be in English!

        Additionally, annotate each triple with:
            - Sentiment (-1 for negative, 0 for neutral, 1 for positive)
            - Polarity (-1 for negation, 0 for neutral/questioning, 1 for affirmation)
            - Certainty (a scale between 0 for uncertain and 1 for certain)
            - Dialogue act (
                0 : "greeting",
                1 : "farewell",
                2 : "negative_reaction",
                3 : "positive_reaction",
                4 : "concern",
                5 : "query",
                6: "other")
            - translated_triple: English version of the triple "subject predicate multitoken_object"

            Ensure that you capture all predicates in an utterance and that they are semantically meaningful.

            Separate multi-word items with an underscore. 

            Save it as a JSON with this format:

            {
            "Conversation ID": "60250de4b",
            "dialogue": [
                {
                "sender": "customer",
                "text": "I can't find my order. It was supposed to arrive yesterday.",
                "triples": [
                    {
                    "subject": "I",
                    "predicate": "can't_find",
                    "object": "my_order",
                    "sentiment": -1,
                    "polarity": -1,
                    "certainty": 1,
                    "dialogue_act": 4
                    },
                    {
                    "subject": "It",
                    "predicate": "was_supposed_to_arrive",
                    "object": "yesterday",
                    "sentiment": -1,
                    "polarity": 1,
                    "certainty": 0.7,
                    "dialogue_act": 4
                    }
                ]
                },
                {
                "sender": "agent",
                "text": "Ich werde Dir damit helfen.",
                "triples": [
                    {
                    "subject": "Ich",
                    "predicate": "werde_helfen",
                    "object": "dir_damit",
                    "sentiment": 1,
                    "polarity": 1,
                    "certainty": 1,
                    "dialogue_act": 3,
                    "translated_triple": "I will_help you_with_that"
                    }
                ]
                }
            ]
            }
        """
        
        user_prompt = f"Analyze the following conversation with ID {conversation_id}: {conversation_text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt_tokens = num_tokens_from_messages(messages)
        total_tokens += prompt_tokens

        if total_tokens >= TOKEN_LIMIT:
            print("Token limit reached, sleeping for a minute...")
            time.sleep(60)
            total_tokens = 0  # Reset token counter after sleep

        else:
            sleep_time = max(0, REQUEST_INTERVAL - (time.time() - start_time))
            time.sleep(sleep_time)

        start_time = time.time()  # Reset start time after sleeping

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=messages,
            temperature=0.0 
        )

        # Print the raw response from GPT-4
        # print("Raw response from GPT-4:", response)
        print(conversation_id)

        result = response['choices'][0]['message']['content']

        # Print the content returned by GPT-4
        # print("Content from GPT-4:", result)

        # Ensure the response is valid JSON
        try:
            extracted_info = json.loads(result)
            return extracted_info, total_tokens
        except json.JSONDecodeError as e:
            error_message = f"JSON decode error for {conversation_id}: {e}"
            log_error(conversation_id, error_message, error_log_path)
            return {}, total_tokens

    except openai.error as e:
        error_message = f"OpenAI API error for {conversation_id}: {e}"
        log_error(conversation_id, error_message, error_log_path)
        return {}, total_tokens

    except ValueError as e:
        error_message = f"Value error for {conversation_id}: {e}"
        log_error(conversation_id, error_message, error_log_path)
        return {}, total_tokens

    except Exception as e:
        error_message = f"Unexpected error for {conversation_id}: {e}"
        log_error(conversation_id, error_message, error_log_path)
        return {}, total_tokens

# Process each conversation and split if necessary
def process_conversations(conversations: Dict[str, pd.DataFrame], error_log_path: str) -> List[Dict]:
    processed_data = []
    total_tokens = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = OrderedDict()
        
        for doc_id, conversation in tqdm(conversations.items(), desc="Processing conversations"):
            if len(conversation) > 60:
                subparts = [conversation.iloc[i:i + 60] for i in range(0, len(conversation), 60)]
            else:
                subparts = [conversation]
            
            for subpart in subparts:
                conversation_text = "\n".join([f"{row['sender']}: {row['text']}" for _, row in subpart.iterrows()])
                formatted_conversation = f"Conversation ID: {doc_id}\n" + conversation_text
                futures[(doc_id, subpart.index[0])] = executor.submit(extract_entities_relationships_and_context, doc_id, formatted_conversation, total_tokens, start_time, error_log_path)
        
        results = OrderedDict()
        for (doc_id, _), future in as_completed(futures.items()):
            try:
                result, total_tokens = future.result()
                if doc_id not in results:
                    results[doc_id] = {"Conversation ID": doc_id, "dialogue": []}
                if result:
                    results[doc_id]["dialogue"].extend(result.get("dialogue", []))
            except Exception as e:
                log_error(doc_id, str(e), error_log_path)
    
    for merged_result in results.values():
        processed_data.append(merged_result)

    return processed_data


# Process train and validation datasets for different language pairs
def main():

    languages = ['en-de', 'en-fr', 'en-nl', 'en-pt']
    datasets = ['test']
    
    for lang in languages:
        for dataset in datasets:
            file_path = f'../../{dataset}/{lang}.csv'
            output_path = f'../graphs/{dataset}/{lang}_new.json'
            error_log_path = f'../graphs/{dataset}/{lang}_error_log.txt'
            
            # Ensure the directories exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load data
            data = load_data(file_path)
            conversations = OrderedDict(list(data.groupby('doc_id')))
            
            # Extract texts
            all_conversations = OrderedDict((doc_id, extract_texts(conversation)) for doc_id, conversation in conversations.items())
            
            # Calculate and print average and maximum dialogue length
            dialogue_lengths = [len(conversation) for conversation in all_conversations.values()]
            average_length = sum(dialogue_lengths) / len(dialogue_lengths) if dialogue_lengths else 0
            max_length = max(dialogue_lengths) if dialogue_lengths else 0
            print(f"{dataset} - {lang}: Average dialogue length: {average_length}")
            print(f"{dataset} - {lang}: Maximum dialogue length: {max_length}")
            
            # Get the processed data
            processed_data = process_conversations(all_conversations, error_log_path)
            
            # Save the processed data to a JSON file
            with open(output_path, 'w') as f:
                json.dump(processed_data, f, indent=2)


if __name__ == "__main__":
    main()
