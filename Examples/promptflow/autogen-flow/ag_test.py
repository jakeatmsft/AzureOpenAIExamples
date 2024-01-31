from promptflow import tool
import autogen
#from autogen import Cache



# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(input1: str) -> str:
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
        },
    )
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
    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # set to True or image name like "python:3" to use docker
        },
    )
    # the assistant receives a message from the user_proxy, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        message=f"""Answer the following: {input1}""",
    )

    output = assistant.last_message(agent=user_proxy)['content']
    #output = assistant.chat_messages()

    return f'{output}'