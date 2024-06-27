

#!/usr/bin/env python
# coding: utf-8

# imports
import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import os  # for environment variables
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import snowflake.connector  # Module for executing queries on snowflake
import warnings
import numpy as np
import json
from ydata_profiling import ProfileReport  # To generate data profiling reports for the data

# Set your API keys and credentials
api_key = "<your chat gpt api key>"
snowflake_user_name = "<your snowflake username>"
snowflake_password = "<your snowflake password>"
client = openai.OpenAI(api_key = api_key)
number_of_sim_files = 5
max_tokens = 8192//200  # Define the maximum token length

max_context_count = 70000

# Define Snowflake connection parameters
snowflake_conn_params = {
    'user': snowflake_user_name,
    'password': snowflake_password,
    'account': '<Snowflake account id>',
    'warehouse': 'compute_wh',
    'database': 'prod_db',
    'schema': 'sales'
}


def get_snowflake_table_column_names(table_name):
    """
    Fetch column names from a Snowflake table.

    Parameters:
    table_name (str): The full name of the table.

    Returns:
    list: A list of column names.
    """
    conn_params = snowflake_conn_params
    conn = snowflake.connector.connect(**conn_params)
    try:
        cursor = conn.cursor()
        sql_query = f"""SELECT column_name
                        FROM prod_db.information_schema.columns
                        WHERE table_name = '{table_name.split('.')[2].upper()}'
                        AND upper(table_schema) = '{table_name.split('.')[1].upper()}'
                        ORDER BY ordinal_position;"""
        cursor.execute(sql_query)
        results = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
    return [row[0] for row in results]


def get_snowflake_table(table_name, order_by_column, where_clause=''):
    """
    Fetch data from a Snowflake table.

    Parameters:
    table_name (str): The full name of the table.
    order_by_column (str): The column to order the results by.
    where_clause (str): Additional where clause for filtering the results.

    Returns:
    pd.DataFrame: DataFrame containing the table data.
    """
    conn_params = snowflake_conn_params
    conn = snowflake.connector.connect(**conn_params)
    try:
        cursor = conn.cursor()
        sql_query = f"SELECT * FROM {table_name} {where_clause} ORDER BY {order_by_column} DESC LIMIT 10000;"
        cursor.execute(sql_query)
        column_names = get_snowflake_table_column_names(table_name)
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=column_names)
    finally:
        cursor.close()
        conn.close()
    return df


def column_metadata_stats(df):
    """
    Generate column metadata statistics for a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    pd.DataFrame: DataFrame containing metadata statistics.
    """
    metadata_df = df.describe(include='all').transpose().reset_index(drop=False)
    metadata_df['Most Frequent Value'] = metadata_df['top'].apply(lambda x: str(x)) + ' : ' + metadata_df['freq'].apply(lambda x: str(x)) + ' times'
    metadata_df['Proportion of Unique Values'] = metadata_df['unique'].fillna(0) / 10000

    all_col_uniques = []
    for col_name in df.columns:
        max_cat = 30
        if len(df[col_name].unique()) < max_cat:
            all_val = ', '.join([str(i) for i in list(df[col_name].unique())])
            all_col_uniques.append(all_val)
        else:
            all_col_uniques.append(f'More than {str(max_cat)}')
    metadata_df['All Distinct Categories'] = all_col_uniques

    return metadata_df[['index', 'Most Frequent Value', 'Proportion of Unique Values', 'All Distinct Categories']]


def extract_sql_from_answer(answer):
    """
    Extracts the SQL query from a GPT response.

    Parameters:
    answer (str): The response from GPT containing the SQL query.

    Returns:
    str: The extracted SQL query.
    """
    pattern = r'```sql\n(.*?)\n```'
    match = re.search(pattern, answer, re.DOTALL)

    if match:
        sql_query = match.group(1).strip()
        return sql_query
    else:
        raise ValueError("No SQL query found in the given answer")


def execute_snowflake_query(gpt_response, attempt=1, max_attempts=3):
    """
    Executes a Snowflake query extracted from a GPT response.

    Parameters:
    gpt_response (str): The GPT response containing the SQL query.
    attempt (int): Current attempt number.
    max_attempts (int): Maximum number of attempts to retry the query.

    Returns:
    pd.DataFrame: DataFrame containing the query results.
    """
    conn_params = snowflake_conn_params
    conn = snowflake.connector.connect(**conn_params)
    cursor = conn.cursor()
    sql_query = extract_sql_from_answer(gpt_response)
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        df = pd.DataFrame(results)
        return df
    except Exception as e:
        if attempt < max_attempts:
            context = str(e) + '\n' + sql_query
            question = 'Find the issue with the query'
            answer = get_answer_from_gpt(api_key, context, question)
            return execute_snowflake_query(answer, attempt + 1, max_attempts)
        else:
            raise Exception(f"Query failed after {max_attempts} attempts: {str(e)}")


def generate_embeddings(api_key, text):
    """
    Generate embeddings for text using OpenAI API.

    Parameters:
    api_key (str): OpenAI API key.
    text (str): Text to generate embeddings for.

    Returns:
    list: List of embeddings.
    """
    openai.api_key = api_key
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding


def split_text_into_chunks(text, max_tokens):
    """
    Split text into chunks that do not exceed the maximum token length.

    Parameters:
    text (str): Text to split.
    max_tokens (int): Maximum number of tokens per chunk.

    Returns:
    list: List of text chunks.
    """
    tokens = text.split(' __ ')
    chunks = [' __ '.join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks


def generate_and_store_embeddings(context_json, api_key, output_path, max_tokens):
    """
    Generate and store embeddings for each file's content.

    Parameters:
    context_json (dict): Dictionary containing file paths and their content.
    api_key (str): OpenAI API key.
    output_path (str): Path to store the embeddings.
    max_tokens (int): Maximum number of tokens per chunk.
    """
    embeddings_data = {}

    for file_path, content in context_json.items():
        try:
            chunks = split_text_into_chunks(content, max_tokens)
            print(len(chunks))
            embeddings = [generate_embeddings(api_key, chunk) for chunk in chunks]
            embeddings_data[file_path] = embeddings
        except Exception as e:
            print(f"Error generating embedding for {file_path}: {e}")
    with open(output_path, 'w') as output_file:
        json.dump(embeddings_data, output_file, indent=4)
    print(f"Embeddings have been written to {output_path}")



def read_json_file(json_path):
    """
    Read the JSON file and get file contents.

    Parameters:
    json_path (str): Path to the JSON file.

    Returns:
    dict: Dictionary containing file contents.
    """
    with open(json_path, 'r') as json_file:
        files_data = json.load(json_file)
    return files_data


def load_embeddings(embeddings_path):
    """
    Load embeddings from the JSON file.

    Parameters:
    embeddings_path (str): Path to the embeddings JSON file.

    Returns:
    dict: Dictionary containing embeddings data.
    """
    with open(embeddings_path, 'r') as embeddings_file:
        embeddings_data = json.load(embeddings_file)
    return embeddings_data


def find_relevant_context(embeddings_data, question_embedding):
    """
    Find the most relevant context based on a business question.

    Parameters:
    embeddings_data (dict): Dictionary containing embeddings data.
    question_embedding (list): List of embeddings for the question.

    Returns:
    list: List of most relevant file paths.
    """
    question_embedding = np.array(question_embedding)
    similarities = {}

    for file_path, embeddings in embeddings_data.items():
        similarity = np.mean([np.dot(question_embedding, np.array(embedding)) for embedding in embeddings])
        similarities[file_path] = similarity

    top_files = sorted(similarities, key=similarities.get, reverse=True)[:number_of_sim_files]
    return top_files


def get_answer_from_gpt(api_key, context, question):
    """
    Call the OpenAI API with the relevant context to get an answer.

    Parameters:
    api_key (str): OpenAI API key.
    context (str): Relevant context.
    question (str): Business question.

    Returns:
    str: Answer from GPT.
    """
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are a sql code generator. 
                                The context contains all the required column name and it's details. Refer to the syntax "COLUMN_NAME"  : "<>" to find the correct column names.
                                Make sure the column value exist before using it in filter. Refer to "" Always use the like condition based on the query shared.
                                Use the context and find accurate table and column names that need to be used to generate the required output"""},
            {"role": "user", "content": "Here is the relevant context:\n" + str(context)},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


def main(question):
    """
    Main function to generate SQL query and execute it based on the provided question.

    Parameters:
    question (str): Business question.

    Returns:
    pd.DataFrame: DataFrame containing the query results.
    """
    # Prepare metadata
    metadata_df = get_snowflake_table("prod_db.information_schema.columns", 'ordinal_position', "WHERE table_name IN ('STORE_SALES', 'DAILY_LOCATIONWISE_SPEND')")
    metadata_df['TABLE_NAME'] = metadata_df['TABLE_CATALOG'] + '.' + metadata_df['TABLE_SCHEMA'] + '.' + metadata_df['TABLE_NAME']

    table_to_profile = ['PROD_DB.SALES.STORE_SALES', 'PROD_DB.SPENDS.DAILY_LOCATIONWISE_SPEND']
    describe_metadata_dfs = []

    for table_name in table_to_profile:
        df = get_snowflake_table(table_name, 'anchor_date')
        df.drop(columns=['ANCHOR_WEEK_START_DATE'], axis=1, inplace=True)
        describe_df = column_metadata_stats(df)
        describe_df['TABLE_NAME'] = table_name
        describe_df.rename(columns={'index': 'COLUMN_NAME'}, inplace=True)
        describe_metadata_dfs.append(describe_df)

    describe_metadata_df = pd.concat(describe_metadata_dfs)
    metadata_merged_df = pd.merge(metadata_df[['TABLE_NAME', 'COLUMN_NAME', 'DATA_TYPE', 'COMMENT']], describe_metadata_df, on=['COLUMN_NAME', 'TABLE_NAME'], how='inner')

    model_context_dict = {}
    for group_key in metadata_merged_df.groupby(['TABLE_NAME']):
        curr_metadata_df = group_key[1]
        curr_metadata_df = curr_metadata_df.drop('TABLE_NAME',axis = 1)
        model_context_dict[group_key[0][0]] = ''
        for key,val in json.loads(curr_metadata_df.transpose().to_json()).items():
            model_context_dict[group_key[0][0]] = model_context_dict[group_key[0][0]] + json.dumps(val) + ' __ '
    # Generate embeddings
    embeddings_path = './context_embeddings.json'
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)


    # Load embeddings and find relevant context
    embeddings_data = load_embeddings(embeddings_path)
    question_embedding = generate_embeddings(api_key, question)
    top_files = find_relevant_context(embeddings_data, question_embedding)

    # Get context and answer from GPT
    context = '\n'.join([f'{i} : {model_context_dict[i]}' for i in top_files])[:max_context_count]
    answer = get_answer_from_gpt(api_key, context, question)
    print("Answer:", answer)

    # Execute the query and return the DataFrame
    df = execute_snowflake_query(answer)
    return df


if __name__ == "__main__":
    question = input("Enter your business question: ")
    result_df = main(question)
    print(result_df)
