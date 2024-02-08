from promptflow import tool
from promptflow.connections import AzureOpenAIConnection

import html
import io
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests
import yfinance as yf

from openai import AzureOpenAI
from openai.types import FileObject
from openai.types.beta import Thread
from openai.types.beta.threads import Run
from openai.types.beta.threads.message_content_image_file import MessageContentImageFile
from openai.types.beta.threads.message_content_text import MessageContentText
from openai.types.beta.threads.messages import MessageFile
from PIL import Image



# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(connection: AzureOpenAIConnection, input1: str, assistant_id: str) -> str:
    def get_stock_price(symbol: str) -> float:
        stock = yf.Ticker(symbol)
        return stock.history(period="1d")["Close"].iloc[-1]


    tools_list = [
        {"type": "code_interpreter"},
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Retrieve the latest closing price of a stock using its ticker symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "The ticker symbol of the stock"}},
                    "required": ["symbol"],
                },
            },
        },
    ]

    # DATA_FOLDER = "data/"

    # def upload_file(client: AzureOpenAI, path: str) -> FileObject:
    #     with Path(path).open("rb") as f:
    #         return client.files.create(file=f, purpose="assistants")

    def call_functions(client: AzureOpenAI, thread: Thread, run: Run) -> None:
        print("Function Calling")
        required_actions = run.required_action.submit_tool_outputs.model_dump()
        print(required_actions)
        tool_outputs = []
        import json

        for action in required_actions["tool_calls"]:
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if func_name == "get_stock_price":
                output = get_stock_price(symbol=arguments["symbol"])
                tool_outputs.append({"tool_call_id": action["id"], "output": output})
            else:
                raise ValueError(f"Unknown function: {func_name}")

        print("Submitting outputs back to the Assistant...")
        client.beta.threads.runs.submit_tool_outputs(thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs)
        

    client = AzureOpenAI(api_key=connection.api_key, api_version='2024-01-01-preview', azure_endpoint=connection.api_base)

    # arr = os.listdir(DATA_FOLDER)
    # assistant_files = []
    # for file in arr:
    #     filePath = DATA_FOLDER + file
    #     assistant_files.append(upload_file(client, filePath))

    # file_ids = [file.id for file in assistant_files]
    

    assistant = client.beta.assistants.retrieve(assistant_id)
    #assistant = client.beta.assistants.create(
    #    name="Portfolio Management Assistant",
    #    instructions="You are a personal securities trading assistant. Please be polite, professional, helpful, and friendly. "
    #    + "Use the provided portfolio CSV file to answer the questions. If question is not related to the portfolio or you cannot answer the question, say, 'contact a representative for more assistance.'"
    #    + "If the user asks for help or says 'help', provide a list of sample questions that you can answer.",
    #    tools=tools_list,
    #    model='gpt-4',
    #    file_ids=file_ids,
    #)

    thread = client.beta.threads.create()
    
    def process_message(content: str) -> None:
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=content)

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="The current date and time is: " + datetime.now().strftime("%x %X") + ".",
        )

        print("processing...")
        while True:
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                return format_messages(messages)
                break
            if run.status == "failed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                return format_messages(messages)
                # Handle failed
                break
            if run.status == "expired":
                # Handle expired
                break
            if run.status == "cancelled":
                # Handle cancelled
                break
            if run.status == "requires_action":
                call_functions(client, thread, run)
            else:
                time.sleep(5)
                
    def format_messages(messages: Iterable[MessageFile]) -> None:
        message_list = []

        # Get all the messages till the last user message
        for message in messages:
            message_list.append(message)
            if message.role == "user":
                break

        # Reverse the messages to show the last user message first
        message_list.reverse()

        # Print the user or Assistant messages or images
        return_msg = []
        for message in message_list:
            for item in message.content:
                # Determine the content type
                #if isinstance(item, MessageContentText):
                return_msg.append(f"{message.role}:\n{item.text.value}\n")
                # elif isinstance(item, MessageContentImageFile):
                #     # Retrieve image from file id
                #     response_content = client.files.content(item.image_file.file_id)
                #     data_in_bytes = response_content.read()
                #     # Convert bytes to image
                #     readable_buffer = io.BytesIO(data_in_bytes)
                #     image = Image.open(readable_buffer)
                #     # Resize image to fit in terminal
                #     width, height = image.size
                #     image = image.resize((width // 2, height // 2), Image.LANCZOS)
                #     # Display image
                #     image.show()
        return message_list
                    
    return process_message(input1)