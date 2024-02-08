from promptflow import tool
from promptflow.connections import AzureOpenAIConnection

import autogen
import json, re
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
#from autogen import Cache

# create a UserProxyAgent instance named "user_proxy"
def has_boxed(string):
    return '\\boxed' in string

def extract_last_boxed_to_newline(s):
    matches = re.findall(r'(\\boxed\{.*?\}.*?)(?=\n|$)', s, re.DOTALL)
    return matches[-1] if matches else None


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(input1: str, connection: AzureOpenAIConnection, modelname: str) -> str:
    # config_list = autogen.config_list_from_json(
    #     "OAI_CONFIG_LIST",
    #     filter_dict={
    #         "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    #     },
    # )
    config_list = [
                    {
                        "model": modelname,
                        "api_key": connection.api_key,
                        "base_url": connection.api_base,
                        "api_type": "azure",
                        "api_version": "2023-07-01-preview"
                    },
                    ]
    # create an AssistantAgent named "assistant"
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={
            "cache_seed": None,  # disable
            "seed": None,  # disable
            "config_list": config_list,  # a list of OpenAI API configurations
            "temperature": 0,  # temperature for sampling
        },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
    )


    # 2. create the MathUserProxyAgent instance named "mathproxyagent"
    # By default, the human_input_mode is "NEVER", which means the agent will not ask for human input.
    mathproxyagent = MathUserProxyAgent(
        name="mathproxyagent", 
        human_input_mode="NEVER",
        is_termination_msg = lambda msg: has_boxed(msg['content']),
        code_execution_config={"use_docker": False},
    )

    #autogen.ChatCompletion.start_logging()

    math_problem = input1
    mathproxyagent.initiate_chat(assistant, problem=math_problem+'' , silent=True,)

    last_response =  assistant.last_message(agent=mathproxyagent)
    last_number = last_response['content']
    #last_number = extract_last_boxed_to_newline(last_response['content'])
    return f'{last_number}'
