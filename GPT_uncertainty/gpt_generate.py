import pdb

from openai import AzureOpenAI
# api_key = 'a1a22832d0a246d08b34909bcf25c6c8'
# api_key="a48f276019654f76b36a00962f1cd4e5"
client = AzureOpenAI(azure_endpoint="https://gehc-openai-prod-eus2.openai.azure.com/",
                     api_key="a1a22832d0a246d08b34909bcf25c6c8",
                     api_version="2023-05-15")

response = client.chat.completions.create(
    model="gpt-4-128k",  # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
    messages=[{
        "role": "system", "content": "Assistant is a large language model trained by OpenAI."
    }, {
        "role": "user", "content": "how to improe focus?"
    }])

print(response.choices[0].message.content)
pdb.set_trace()
