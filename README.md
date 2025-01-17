#### 說明

這是 20240713 在 Chatbot 社群分享的內容，帶大家快速建立 LangServe 的來給 Chatbot 串接用。

#### 安裝環境

1. `poetry new chatbotdemo`
2. `cd chatbotdemo`
3. `poetry add langchain langchain-openai qdrant-client langchain-community pypdf`
4. 記得要讓你的 Python 虛擬環境是 3.11 以上： `poetry env use 3.11 `
####  設定好你的 Azure OpenAI 和 Qdrant

1. 取得 Azure OpenAI 的 key 和 endpoint 
2. 如果你是 OpenAI 的使用者要記得改程式碼
3. 取得 Qdrant 的 endpoint 和 api-key
4. 取得政府托育補助的 pdf： https://www.sfaa.gov.tw/SFAA/Pages/ashx/File.ashx?FilePath=~/File/Attach/10802/File_191388.pdf
 
#### RAG 測試

`chatbotdemo/rag.py`


```
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

loader = PyPDFLoader("./qa.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(    
    chunk_size=500,
    chunk_overlap=100)

chunks = splitter.split_documents(docs)

embeddings_model = AzureOpenAIEmbeddings(
    api_key="xx",
    azure_deployment="text-small", 
    openai_api_version="2024-02-15-preview",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
)


qdrant = Qdrant.from_documents(
    chunks,
    embeddings_model,
    url="https://xx.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="xx-xx-xx",
    collection_name="subsidy_qa",
    force_recreate=True,
)


retriever = qdrant.as_retriever(search_kwargs={"k": 3})


model = AzureChatOpenAI(
    api_key="xx",
    openai_api_version="2024-02-15-preview",
    azure_deployment="gpt-4o",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""請回答依照 context 裡的資訊來回答問題:
<context>
{context}
</context>
Question: {input}""")


document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "請問第二胎補助加發多少，共為多少錢？"})

print(response["answer"])


```

`poetry shell`
`python chatbotdemo/rag.py`


#### 改寫成 LangServe


`poetry add langchain-cli` 

在安裝 langchain-cli 時，如果會跳出 uvicorn 要是 0.23.2 的問題，這時候我們去 pyproject. Toml 裡，把 uvicorn 那裡，設定為 uvicorn = "0.23.2"，fastapi = "^0.110.0"，再 `poetry update`

`langchain app new langserveapp`

先 exit 現在的虛擬環境，然後 `cd langserveapp`，進去後再一次 poetry install，然後再把該安裝的套件安裝上。

可能需要去 toml 檔把 pydantic 改成 `pydantic = ">2"`。因為現在 LangServe 需要 1 版 pydantic ，可是其他的套件需要 2 版以上的 pydantic。

再 `poetry shell` 進到我們這個 langserve 的虛擬環境中。


檔名：`rag/rag_chain.py`

```
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.pydantic_v1 import BaseModel



embeddings_model = AzureOpenAIEmbeddings(
    api_key="xx",
    azure_deployment="text-small", 
    openai_api_version="2024-02-15-preview",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
)

client = QdrantClient(
	url="https://xx.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="xx")
collection_name = "subsidy_qa"
qdrant = Qdrant(client, collection_name, embeddings_model)

retriever = qdrant.as_retriever(search_kwargs={"k": 3})

model = AzureChatOpenAI(
    api_key="xx",
    openai_api_version="2024-02-15-preview",
    azure_deployment="gpt-4o",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""請回答依照 context 裡的資訊來回答問題:
<context>
{context}
</context>
Question: {input}""")


document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Add typing for input
class Question(BaseModel):
    input: str


rag_chain = retrieval_chain.with_types(input_type=Question)

```

修改內容：`rag/__init__.py`

```
from rag.rag_chain import rag_chain

__all__ = ["rag_chain"]
```

修改內容：`server.py`

```
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag import rag_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

```

啟動 LangServe `langchain serve`


Invoke 這個路由就是把 API 叫起來的方式，你就可以用這個 API 整合到你要串 LINE 的 webhook 裡。
```
curl --location --request POST 'http://localhost:8000/rag/invoke' \
--header 'Content-Type: application/json' \
--data-raw '{
    "input": {
        "input": "父母離婚，該由誰申請補助"
    }
}'

```

可以進到 play ground  `http://127.0.0.1:8000/rag/playground/` 來試玩

#### 改寫與串接到 LINE
`poetry add line-bot-sdk httpx`

改寫 `server.py` 如下：
```

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag import rag_chain
from pydantic import BaseModel
import json
from linebot.v3 import (
    WebhookParser
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    AsyncApiClient,
    AsyncMessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)
import httpx

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chain, path="/rag")

LINE_ACCESS_TOKEN = 'xx'
CHANNEL_SECRET = 'xx'

configuration = Configuration(access_token=LINE_ACCESS_TOKEN)
async_api_client = AsyncApiClient(configuration)
line_bot_api = AsyncMessagingApi(async_api_client)
parser = WebhookParser(CHANNEL_SECRET)


class CallbackRequest(BaseModel):
    events: list
    destination: str

class RagInvokeRequest(BaseModel):
    input: dict

@app.post("/callback")
async def callback(request: Request):

    x_line_signature = request.headers['X-Line-Signature']

    body = await request.body()
    body = body.decode("utf-8")
    user_id = ""
    try:
        user_id = json.loads(body)["events"][0]["source"]["userId"]
        events = parser.parse(body, x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue

        async def send_loading_animation(chat_id):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'https://api.line.me/v2/bot/chat/loading/start',
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {LINE_ACCESS_TOKEN}'
                    },
                    json={
                        'chatId': chat_id,
                        'loadingSeconds': 10
                    }
                )
                return response
        
        async def call_rag_invoke(input_text):
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    'http://localhost:8000/rag/invoke',
                    json={'input': {'input': input_text}},
                    headers={'Content-Type': 'application/json'}
                )
                return response.json()
        
        await send_loading_animation(user_id)

        response = await call_rag_invoke(event.message.text)
    
        reply_text = response['output']['answer']
        print("Reply text: " + reply_text)
        await line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )
    return 'OK'

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


```


`langchain server` 然後 `ngrok http 8000`，再去 LINE developer 設定 webhook，記得要放上正確的路由，以本範例為例是要在 ngrok 網址最後再加上 `/callback`。

