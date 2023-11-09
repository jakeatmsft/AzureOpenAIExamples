from promptflow import tool
import pandas as pd
from promptflow import tool
import subprocess
import re

def score(csvfile:str, program: str) -> str:
    program = "import pandas as pd\ndf = pd.read_csv('{}')\n".format(csvfile)+program
    program += '\nprint(ans)'
    program = program.replace('\\n', '\n')
    print(program)
    result = subprocess.run(['python', '-c', program], capture_output=True)
    print(result)
    ans =str(result.stdout)
    return ans
# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(csvfile:str, input_program: str) -> str:
  result = score(csvfile, input_program)
  #return_result = re.sub(r'[^\d.]+', '', result)
  return result
