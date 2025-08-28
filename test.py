# # To run this code you need to install the following dependencies:
# # pip install google-genai

# import base64
# import os
# from google import genai
# from google.genai import types


# def generate():
#     client = genai.Client(
#         api_key="AIzaSyAYUEk1AWUnqC7luWnAxNetqcdxvNy_T_s")

#     model = "gemini-2.5-flash"
#     contents = [
#         types.Content(
#             role="user",
#             parts=[
#                 types.Part.from_text(text="Based on you True knowledge, not facts...you think astrology is real..?"),
#             ],
#         ),
#     ]
#     tools = [
#         types.Tool(googleSearch=types.GoogleSearch(
#         )),
#     ]
#     generate_content_config = types.GenerateContentConfig(
#         thinking_config = types.ThinkingConfig(
#             thinking_budget=-1,
#         ),
#         tools=tools,
#     )

#     for chunk in client.models.generate_content_stream(
#         model=model,
#         contents=contents,
#         config=generate_content_config,
#     ):
#         print(chunk.text, end="")

# if __name__ == "__main__":
#     generate()







# To run this code you need to install the following dependencies:
# pip install google-genai

# import os
# from google import genai

# client = genai.Client(api_key="AIzaSyB6bvdq2cYkbQlAfPv-NN4VpqxVqqoTCB4")

# # Simple test prompt
# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Explain what you see in this PDF?"
# )

# print(response.text)

# uploaded_file = client.files.upload(file="uploads/ad7ad480-2854-4e81-b1df-2f73e678934c_ATS_Resume.pdf")



import os
from google import genai
from google.genai import types

# It's recommended to load the API key from an environment variable for security
# genai.configure(api_key=os.environ["GEMINI_API_KEY"]) # You can set this up
client = genai.Client(api_key="AIzaSyB6bvdq2cYkbQlAfPv-NN4VpqxVqqoTCB4")

# Path to your PDF file
pdf_file_path = "uploads/ad7ad480-2854-4e81-b1df-2f73e678934c_ATS_Resume.pdf"

# 1. Upload the PDF file using the File API
print(f"Uploading file: {pdf_file_path}...")
uploaded_file = client.files.upload(file=pdf_file_path)
print(f"Uploaded file: {uploaded_file.uri}") # The URI can be used to reference the file in generate_content

# 2. Construct the content list with both the text prompt and the uploaded file reference
# The 'contents' parameter expects a list of parts.
# You can include text parts and file parts (referenced by the uploaded_file object).
contents = [
    uploaded_file,  # Reference to the uploaded PDF file
    "Explain what you see in this PDF and summarize its key contents. [1, 3]" # Your text message
]

# 3. Call generate_content with the model and the multimodal contents
print("Generating content from the PDF and prompt...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents
)

print("\nResponse from Gemini:")
print(response.text)

# Optional: Delete the uploaded file if it's no longer needed (files are automatically deleted after 48 hours)
# client.files.delete(name=uploaded_file.name)
# print(f"Deleted uploaded file: {uploaded_file.name}")
