


from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re

import pandas as pd
import os


# Get the base directory of the Django project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the absolute path to the CSV file
CSV_FILE_PATH = os.path.join(BASE_DIR, 'chatbot', 'dataframe.csv')

# Read the CSV file
df = pd.read_csv(CSV_FILE_PATH)


# Replace NaN values with a space
df.fillna(' ', inplace=True)
# Load a pre-trained BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def find_most_similar_text(message, df, column_names=['col1', 'col2', 'col3']):
    # Encode the input message
    message_embedding = model.encode(message, convert_to_tensor=True)

    # Encode the concatenated content of specified columns for each row
    concatenated_texts = [' '.join([str(df[col].iloc[i]) for col in column_names]) for i in range(len(df))]
    df_embeddings = model.encode(concatenated_texts, convert_to_tensor=True)

    # Calculate cosine similarity between the input message and all rows with concatenated texts
    similarities = util.pytorch_cos_sim(message_embedding, df_embeddings)[0]

    # Find the index of the row with the highest similarity
    most_similar_index = similarities.argmax().item()

    # Return the most similar text and its position in the DataFrame
    most_similar_text = ' '.join([str(df[col].iloc[most_similar_index]) for col in column_names])
    position = most_similar_index

    return most_similar_text, position

def extract_link_from_figures(figures_content):
    # Use regular expression to find a link in the figures content
    link_pattern = re.compile(r'https?://\S+')
    match = link_pattern.search(figures_content)
    return match.group() if match else None

def ask_for_link(user_message, df=df):
    result_text, result_position = find_most_similar_text(user_message, df, column_names=['Titre', 'Clean_Paragraphe', 'Bart Summary'])
    

    if result_position is not None:
        figures_content = df['figures'].iloc[result_position]
        print("Content of 'figures' column:", figures_content)

        # Extract link from figures content
        link = extract_link_from_figures(figures_content)
        if link:
            print("Link found:", link)
        else:
            print("No link or image available for this content.")
    
    return link




