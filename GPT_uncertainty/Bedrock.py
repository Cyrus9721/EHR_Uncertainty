OPENAI_API_BASE_URL = "https://lmaas-beta.ai.gehealthcare.com/bedrock"
MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"

import os 
os.environ["APP_CLIENT_ID"] = "RkiNbjObr_28nIQivOPwfPoxqIga"
os.environ["APP_CLIENT_SECRET"] = "fkPa4lkLyhD1ib0IE3LGuu7Pfpka"
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from llm_idam_token_generator.idam_token_generator import get_llm_access_token
chat = ChatOpenAI(
    model=MODEL,
    temperature=0,
    openai_api_key=f"{get_llm_access_token()}",
    openai_api_base=OPENAI_API_BASE_URL
)
template = """
Question: {question}
Answer: Let's think step by step.
"""
print(f"Token is : {get_llm_access_token()}")
prompt = PromptTemplate.from_template(template)
llm_chain = prompt | chat
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
response = llm_chain.invoke(question)
print(response)
# Below is the sample response - 
"""
content="Okay, let's think through this step-by-step:
1) First, we need to find out the year Justin Bieber was born. A quick online search shows that Justin Bieber was born on March 1, 1994.
2) So we need to determine which NFL team won the Super Bowl for the 1993 season (since the Super Bowl is played early in the calendar year after the regular season).
3) The Super Bowl for the 1993 NFL season was Super Bowl XXVIII. It was played on January 30, 1994 at the Georgia Dome in Atlanta, Georgia.
4) In that Super Bowl, the Dallas Cowboys defeated the Buffalo Bills by a score of 30-13.
5) Therefore, the NFL team that won the Super Bowl in the year Justin Bieber was born (1994) was the Dallas Cowboys." 
additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 195, 'prompt_tokens': 39, 'total_tokens': 234}, 
'model_name': 'anthropic.claude-3-sonnet-20240229-v1:0', 'system_fingerprint': 'fp', 'finish_reason': 'stop', 'logprobs': None} 
id='run-f73fb4ec-56a0-482c-b2ba-c0e1be08ace9-0' usage_metadata={'input_tokens': 39, 'output_tokens': 195, 'total_tokens': 234}
"""