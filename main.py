import dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatMessagePromptTemplate,
)
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory

dotenv.load_dotenv()

chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("message.json"),
    llm=chat,
    memory_key="messages",
    return_messages=True,
)

prompt = ChatMessagePromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)


while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])
