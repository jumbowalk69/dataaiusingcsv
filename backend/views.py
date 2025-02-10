import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import openai
import json

# Load the Excel data
EXCEL_FILE_PATH = "PO.csv"
dataframe = pd.read_csv(EXCEL_FILE_PATH)

openai.api_key = ""

# Column definitions (hardcoded meanings)
COLUMN_DEFINITIONS = {
    "PO Number": "Purchase Order Number, which uniquely identifies a purchase order.",
    "Company": "The company issuing or receiving the purchase order.",
    "Year": "The year when the transaction occurred.",
    "Amount": "The monetary value of the purchase order.",
    "Project": "The project associated with the purchase order.",
    "Quantity": "The number of items or units in the purchase order.",
    "Pur Group Desc": "The department under which the purchase order falls.",
    "Vendor Desc": "It is the name of the vendor."
}

# Column aliases (user-friendly terms mapped to actual column names)
COLUMN_ALIASES = {
    "purchase order number": "PO Number",
    "company name": "Company",
    "transaction year": "Year",
    "total amount": "Amount",
    "project name": "Project",
    "unit quantity": "Quantity",

}

# Predefined sets of questions
PREDEFINED_QUESTIONS = {
    "most_po_company": "Which company has the most number of Purchase Orders?",
    "highest_amount_po": "Which purchase order has the highest amount?",
    "project_po_count": "How many purchase orders are associated with each project?",
    "yearly_po_summary": "Provide a yearly summary of purchase orders."
}


def resolve_column_aliases(query):
    """
    Resolve user-friendly names in the query to actual column names.
    """
    for alias, column in COLUMN_ALIASES.items():
        if alias.lower() in query.lower():
            query = query.lower().replace(alias.lower(), column)
    return query


def map_query_to_predefined(query):
    """
    Map the user query to a predefined question for better handling.
    """
    for key, question in PREDEFINED_QUESTIONS.items():
        if question.lower() in query.lower():
            return key
    return None


def filter_and_analyze(query):
    """
    Basic filtering and analysis using pandas.
    """
    try:
        resolved_query = resolve_column_aliases(query)

        if "year" in resolved_query:
            year = query.get("year")
            filtered_data = dataframe[dataframe['Year'] == year]
            result = filtered_data.to_dict(orient='records')
        elif "project" in resolved_query:
            project_name = query.get("project")
            filtered_data = dataframe[dataframe['Project'].str.contains(project_name, case=False, na=False)]
            result = filtered_data.to_dict(orient='records')
        else:
            result = {"message": "Unsupported query type"}

        if len(result) > 0:
            summary = {
                "total_entries": len(result),
                "total_amount": dataframe["Amount"].sum() if "Amount" in dataframe.columns else "N/A",
                "unique_projects": dataframe["Project"].nunique() if "Project" in dataframe.columns else "N/A"
            }
            return {"data": result, "summary": summary}

        return {"message": "No matching data found"}
    except Exception as e:
        return {"error": str(e)}


def analyze_with_openai(query, dataframe_summary, column_definitions, predefined_question_key=None):
    """
    Use OpenAI API for concise query interpretation and analysis, with additional context.
    """
    try:
        # Create a brief and focused prompt
        prompt = f"""
        You are an expert data analyst. A user asked the following question: {query}.
        Here is a summary of the dataset: {dataframe_summary}.
        Here are column definitions for better understanding: {column_definitions}.
        """

        # Add predefined question context, if available
        if predefined_question_key:
            prompt += f" This query corresponds to the predefined analysis: '{PREDEFINED_QUESTIONS[predefined_question_key]}'."

        prompt += " Provide a brief and precise answer to their query."

        # Send the prompt to OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data analysis assistant providing concise answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300  # Limit response length
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error analyzing with OpenAI: {str(e)}"


@csrf_exempt
def log_query(request):
    """
    API endpoint for receiving queries and returning analysis results with OpenAI support.
    """
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            user_query = body.get("query", "")

            # Resolve column aliases in the query
            resolved_query = resolve_column_aliases(user_query)

            # Match query with predefined questions
            predefined_question_key = map_query_to_predefined(resolved_query)

            # Perform basic filtering and analysis
            basic_analysis = filter_and_analyze(resolved_query)

            # Generate a dataset summary for OpenAI
            dataframe_summary = {
                "total_entries": len(dataframe),
                "total_columns": len(dataframe.columns),
                "columns": list(dataframe.columns),
                "sample_data": dataframe.head(20).to_dict(orient='records')
            }

            # Use OpenAI for advanced analysis
            openai_analysis = analyze_with_openai(
                resolved_query,
                dataframe_summary,
                COLUMN_DEFINITIONS,
                predefined_question_key
            )

            # Combine results
            response = {
                "basic_analysis": basic_analysis,
                "openai_analysis": openai_analysis
            }
            return JsonResponse(response, safe=False)
        except Exception as e:
            return JsonResponse({"error": f"Failed to process request: {str(e)}"}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=400)


# import os
# import openai
# import pandas as pd
# from pandasai import SmartDataframe
# from pandasai.llm import OpenAI
# from pandasai.helpers.openai_info import get_openai_callback
# from langchain_community.chat_models import ChatOpenAI
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# import spacy
# import re
#
# # Set the OpenAI API key
# os.environ['OPENAI_API_KEY'] = ''
# openai.api_key = os.getenv('OPENAI_API_KEY')
#
# # Load the CSV file
# CSV_FILE_PATH = "PO.csv"
# try:
#     df = pd.read_csv(CSV_FILE_PATH)
#     print("CSV file loaded successfully.")
# except Exception as e:
#     print(f"Error loading CSV file: {e}")
#     df = pd.DataFrame()  # Fallback empty DataFrame
#
# # Automatically identify the DataFrame headers and cache them
# df_columns = df.columns.tolist()
# print(f"Detected DataFrame headers: {df_columns}")
#
# # Initialize spaCy for NLP processing (minimize resource usage by using smaller model)
# nlp = spacy.load("en_core_web_sm")
#
# # Initialize LangChain agent (disabled verbose mode to optimize performance)
# chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)
# agent = create_pandas_dataframe_agent(chat, df, verbose=False, allow_dangerous_code=True)
#
#
# def preprocess_query(query):
#     """
#     Function to preprocess and clean the query using NLP techniques.
#     The focus is on reducing the query length and complexity to minimize token usage.
#     """
#     # Tokenization and lowercasing using spaCy (with minimal processing)
#     doc = nlp(query.lower())
#
#     # Remove stopwords and punctuation for brevity
#     cleaned_query = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
#
#     # Further optimizations: Consider truncating or simplifying long queries if they are too verbose.
#     # Optional: Add custom logic for trimming long queries to limit token count
#
#     return cleaned_query[:100]  # Limiting the length to 100 characters for optimization
#
#
# @csrf_exempt
# def log_query(request):
#     if request.method == 'POST':
#         try:
#             # Parse the request body
#             data = json.loads(request.body)
#             query = data.get('query')
#
#             if not query:
#                 return JsonResponse({'status': 'failed', 'error': 'Query is missing'}, status=400)
#
#             print(f"Received query: {query}")
#
#             # Preprocess the query to reduce unnecessary complexity and token usage
#             query = preprocess_query(query)
#             print(f"Processed query: {query}")
#
#             # Set up PandasAI with OpenAI (use cached API key and avoid redundant setups)
#             llm = OpenAI(api_key=openai.api_key)
#             smart_df = SmartDataframe(df, config={"llm": llm, "conversational": False})
#
#             # Run the query using the uploaded data
#             with get_openai_callback() as cb:
#                 response = smart_df.chat(query)
#                 print(f"Response: {response}")
#
#                 # Check if response is a DataFrame and convert it to JSON-serializable format
#                 if isinstance(response, pd.DataFrame):
#                     response_json = response.to_dict(orient='records')  # Convert to list of dictionaries
#                 else:
#                     response_json = response  # Keep it as is if it's already JSON-serializable
#
#                 return JsonResponse({'status': 'success', 'response': response_json}, status=200)
#
#         except Exception as e:
#             print(f"Error during query execution: {e}")
#             return JsonResponse({'status': 'failed', 'error': str(e)}, status=500)
#
#     return JsonResponse({'status': 'failed', 'error': 'Invalid request method'}, status=405)


def show_df(request):
    print(f"Received request: {request.method}")  # Log the request method
    if request.method == 'GET':
        df_cleaned = df.drop(columns=['Update_Details'], errors='ignore')
        # Convert DataFrame to a list of dictionaries
        response = df_cleaned.to_dict(orient='records')

        # Replace NaN values with empty strings
        for record in response:
            for key, value in record.items():
                if isinstance(value, float) and pd.isna(value):  # Check for NaN
                    record[key] = ""  # Replace with empty string

        return JsonResponse({'status': 'success', 'response': response}, status=200)

    return JsonResponse({'status': 'failed', 'error': 'Invalid request method'}, status=405)


# @csrf_exempt
# def log_query(request):
#     if request.method == 'POST':
#         try:
#             # Parse the request body
#             data = json.loads(request.body)
#             query = data.get('query')
#
#             if not query:
#                 return JsonResponse({'status': 'failed', 'error': 'Query is missing'}, status=400)
#
#             print(f"Received query: {query}")
#
#             # Run the query using the LangChain agent
#             response = agent.run(query)
#             print(f"Agent response: {response}")
#
#             # Print the number of tokens utilized for both input and output
#             input_tokens = chat.get_num_tokens(query)
#             output_tokens = chat.get_num_tokens(response)
#             total_tokens = input_tokens + output_tokens
#
#             print(f"Input tokens: {input_tokens}")
#             print(f"Output tokens: {output_tokens}")
#             print(f"Total tokens: {total_tokens}")
#
#             # Return the response back to the frontend
#             return JsonResponse({'status': 'success', 'response': response, 'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens}, status=200)
#
#         except Exception as e:
#             print(f"Error during query execution: {e}")
#             return JsonResponse({'status': 'failed', 'error': str(e)}, status=500)
#
#     return JsonResponse({'status': 'failed', 'error': 'Invalid request method'}, status=405)


#---------------------------------------------------------------------------------------------------


# import os
# import pandas as pd
# from pandasai import SmartDataframe
# from langchain_groq.chat_models import ChatGroq
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
#
# # Load the CSV file
# csv_file = 'Format & FS of detail PO Report 4 (1).csv'
# try:
#     df = pd.read_csv(csv_file)
#     print("CSV file loaded successfully.")
# except Exception as e:
#     print(f"Error loading CSV file: {e}")
#     df = pd.DataFrame()  # Create an empty DataFrame as a fallback
#
# # Initialize the LLM
# llm = ChatGroq(model_name="llama3-70b-8192", api_key="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")
# df = SmartDataframe(df, config={"llm": llm})
#
# def show_df(request):
#     print(f"Received request: {request.method}")  # Log the request method
#     if request.method == 'GET':
#         df_cleaned = df.drop(columns=['Update_Details'], errors='ignore')
#         # Convert DataFrame to a list of dictionaries
#         response = df_cleaned.to_dict(orient='records')
#
#         # Replace NaN values with empty strings
#         for record in response:
#             for key, value in record.items():
#                 if isinstance(value, float) and pd.isna(value):  # Check for NaN
#                     record[key] = ""  # Replace with empty string
#
#         return JsonResponse({'status': 'success', 'response': response}, status=200)
#
#     return JsonResponse({'status': 'failed', 'error': 'Invalid request method'}, status=405)
#
# @csrf_exempt
# def log_query(request):
#     if request.method == 'POST':
#         try:
#             # Parse the request body
#             data = json.loads(request.body)
#             query = data.get('query')
#
#             if not query:
#                 return JsonResponse({'status': 'failed', 'error': 'Query is missing'}, status=400)
#
#             print(f"Received query: {query}")
#
#             # Run the query using the SmartDataframe
#             response = df.chat(query)
#             print(f"Response: {response}")
#
#             # Convert the response to a JSON-serializable format
#             if isinstance(response, pd.DataFrame):
#                 response = response.to_dict(orient='records')
#
#             # Return the response back to the frontend
#             return JsonResponse({'status': 'success', 'response': response}, status=200)
#
#         except Exception as e:
#             print(f"Error during query execution: {e}")
#             return JsonResponse({'status': 'failed', 'error': str(e)}, status=500)
#
#     return JsonResponse({'status': 'failed', 'error': 'Invalid request method'}, status=405)




#-------------------------------------------------------------------------------------------------
