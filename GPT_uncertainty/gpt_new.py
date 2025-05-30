from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

import truststore
truststore.inject_into_ssl()
 

os.environ["APP_CLIENT_ID"] = "zizhang-chen-research-app"
os.environ["APP_CLIENT_SECRET"] = "cQPcueFP7tDrimbf8NW2GAHcHeQa"
from llm_idam_token_generator.idam_token_generator import get_idam_token
# OpenAI Endpoint details
OPENAI_ENDPOINT = "https://openai-llm-frontdoor-hma7evbthrd4cugn.a01.azurefd.net"
# OPENAI_DEPLOYMENT_MODEL = "gpt-4o"
OPENAI_DEPLOYMENT_MODEL = "gpt4_32k"
OPENAI_AZURE_API_VERSION = "2023-12-01-preview"
OPENAI_TYPE="azure"



llm = AzureChatOpenAI(
    api_key="xxx",  # This is not playing any role, but required as per OpenAI sdk. So any random could be passed.
    azure_endpoint=OPENAI_ENDPOINT,
    deployment_name=OPENAI_DEPLOYMENT_MODEL,
    openai_api_version=OPENAI_AZURE_API_VERSION,
    default_headers={
        'Authorization': f'Bearer {get_idam_token()}',
        'Content-Type': 'application/json'
    }
)

prompt = ChatPromptTemplate.from_template("what is the city {person} is from?")
chain = prompt | llm
print(chain.invoke({"person": "Narendra modi"}))