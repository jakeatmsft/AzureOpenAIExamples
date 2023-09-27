from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from transformers import AutoTokenizer, pipeline
import torch
from flask import Flask, render_template, request
import re

# choose model based on your hardware
model = 'tiiuae/falcon-7b-instruct'
# model = 'tiiuae/falcon-40b-instruct'

# load a tokenizer from a pretrained model using the Hugging Face AutoTokenizer class
# and the from_pretrained method is used to retrieve the tokenizer associated with the specified model
# to which the tokenizer is responsible for processing text inputs and converting them into numerical 
# representations suitable for input to the model
print('loading model')
tokenizer = AutoTokenizer.from_pretrained(model)

# the pipeline function returns a callable object that can be used to generate text using 
# the specified model and parameters
print('loading pipeline')
pipeline = pipeline(
    'text-generation',  # the task for the pipeline
    model=model,  # the pretrained model to use
    tokenizer=tokenizer,  # the tokenizer for preprocessing inputs
    torch_dtype=torch.bfloat16,  # the data type for torch tensors
    trust_remote_code=True,  # flag to trust remote code (e.g., when using remote models)
    device_map='auto',  # the device to run the pipeline on (GPU or CPU)
    max_length=20000,  # the maximum length of generated text
    do_sample=True,  # flag indicating whether to use sampling for text generation
    top_k=10,  # the number of highest probability tokens to consider for sampling
    num_return_sequences=1,  # the number of sequences to generate
    eos_token_id=tokenizer.eos_token_id  # the token ID representing the end of a text sequence
)

# the HuggingFacePipeline instance llm is created with the specified pipeline and model_kwargs and 
# the llm object can then be used to generate text based on the configured pipeline and model parameters
# create an instance of the HuggingFacePipeline class
print('loading llm')
llm = HuggingFacePipeline(
    pipeline=pipeline,  # the text generation pipeline to use
    model_kwargs={'temperature': 0}  # temperature is a common parameter used in text generation models to 
                                     # control the randomness of the generated output and the higher 
                                     # temperature values (e.g., 1.0) lead to more diverse and creative 
                                     # output, while lower values (e.g., 0.5) make the output more
                                     # focused and deterministic
)

# define the template for the prompt
template = """
You are an intelligent chatbot. Take careful consideration to context of the question and answer appropriately.
Question: {question}
Answer:"""

# define a template for the prompt to be used in the LLMChain instance and the prompt template allows for 
# customization of the prompt message and dynamic insertion of input variables and
# the template variable stores a multi-line string that serves as the template for the prompt and it provides a general 
# message for the chatbot and defines the format for presenting the question and answer and the PromptTemplate class is 
# instantiated with two arguments which are template, the template string defined earlier, which serves as the base 
# structure for the prompt and input_variables, a list of input variables used in the template and in this case, we 
# have only one variable, 'question', which represents the user's input question to which the PromptTemplate object 
# prompt is created, which can be used within the LLMChain instance to generate prompts dynamically based on user input
# and by using prompt templates, you can create flexible and customizable prompts that adapt to the user's specific 
# input, making the conversation more engaging and interactive
# create a prompt template
prompt = PromptTemplate(
    template=template,  # the template string for the prompt
    input_variables=['question']  # the list of input variables used in the template
)

# create an instance of the LLMChain class
llm_chain = LLMChain(
    prompt=prompt,   # the prompt template for generating prompts
    llm=llm  # the HuggingFacePipeline instance for text generation
)


def remove_angle_brackets(text):
    """
    Removes angle brackets and their contents from the given text.
    
    Args:
        text (str): The input text from which angle brackets and their contents need to be removed.
    
    Returns:
        str: The modified text with angle brackets and their contents removed.
    """
    return re.sub(r'<[^>]*>', '', text)


# init the Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Renders the home page and handles form submission.

    If the request method is POST, it retrieves the question from the form,
    generates a response using the LLMChain, removes angle brackets from the response,
    and renders the updated index.html template with the response.
    
    If the request method is GET, it renders the index.html template.

    Returns:
        str: The rendered HTML template for the home page.
    """
    if request.method == 'POST':
        question = request.form['question']
        response = llm_chain.run(question)
        response = remove_angle_brackets(response)
        return response
    return "POST form[question]"


if __name__ == '__main__':
    # check if CUDA is available and being used
    if torch.cuda.is_available() and torch.cuda.current_device() != -1:
        print('CUDA is being used.')
    else:
        print('CUDA is not being used.')
    print('running server')
    # run app
    app.run(host='0.0.0.0', port=5000, debug=False)