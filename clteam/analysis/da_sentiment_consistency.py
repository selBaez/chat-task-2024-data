import json

# - Sentiment (-1 for negative, 0 for neutral, 1 for positive)
# - Dialogue act (
#           0 : "greeting",
#           1 : "farewell",
#           2 : "negative_reaction",
#           3 : "positive_reaction",
#           4 : "concern",
#           5 : "query",
#           6 : "other")


# Load data
with open('selea/chat-task-2024-data/clteam/graphs/valid/en-de.json', 'r') as file:
    data = json.load(file)

# Function to check consistency
def check_consistency(data):
    dialogue_act_sentiment = {
        0: [0, 1],  # greeting: neutral or positive sentiment
        1: [0, 1],  # farewell: neutral or positive sentiment
        2: [-1],  # negative_reaction: negative sentiment
        3: [1],  # positive_reaction: positive sentiment
        4: [-1, 0],  # concern: negative or neutral sentiment
        5: [-1, 0, 1],  # query: any sentiment
        6: [-1, 0, 1]  # other: any sentiment
    }
    
    inconsistent_triples = []
    
    for conv in data:
        for message in conv['dialogue']:
            for triple in message['triples']:
                sentiment = triple['sentiment']
                dialogue_act = triple['dialogue_act']
                
                if sentiment not in dialogue_act_sentiment[dialogue_act]:
                    inconsistent_triples.append(triple)
    
    return inconsistent_triples

# Check consistency
inconsistent_triples = check_consistency(data)

# Print inconsistent triples
if inconsistent_triples:
    print("Inconsistent triples found:")
    for triple in inconsistent_triples:
        print(triple)
else:
    print("All triples are consistent.")
