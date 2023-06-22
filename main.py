import os
import config
import secret_config
import logging
import redis
import requests
from fastapi import (
    FastAPI,
    Request,
    Response,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import chromadb

# Bootstrapping the application
## Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    filename=os.path.join('log', 'main.log'),
)
## Redis
redis_connection = redis.Redis(
    password=secret_config.REDIS_PASSWORD,
    decode_responses=True,
    encoding='utf-8',
)
## Langchain
os.environ['OPENAI_API_KEY'] = secret_config.OPENAI_API_KEY
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
### Panelist
panelist_db = PGVector(
    embedding_function=embeddings,
    collection_name=config.PANELIST_LANGCHAIN_VECTOR_COLLECTION_NAME,
    connection_string=secret_config.PANELIST_PGVECTOR_CONNECTION_STRING,
)
panelist_retriever = panelist_db.as_retriever()
panelist_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=panelist_retriever,
)
panelist_prompt_template_text = """
I want you to act as a tech startup competition panelist or judge.
Adopt the following personality and characteristics:
{personality}

Do an evaluation of the overall pitch and business model and highlight gaps. If there are big gaps, state outright that the pitch needs improvement.

Write an evaluation and score or grade (from 0-10) the pitch based on these items:

- Business Model (esp. value proposition, customer segments, revenue stream, customer channels) (score 1-10 depending on clarity and completeness)
- What sets the business apart? What are the key differentiators? (Score 1-10 depending on clarity and soundness.)
- Market potential: TAM, SAM, SOM. Is it clear that the founding team knows what these are? Are the SOM targets appropriate for the geographic area cited (if any)? Are the numbers stated in monetary amounts, or simple customer counts? Note that these should be in monetary values. (score 1-10 depending on completeness and soundness)
- Co-founders and founder-market fit. Who are the co-founders? What is the role of each? Is there a Co-founders' Agreement in place? Is there someone in the team who can build this? (Score 1-10 depending on clarity and competitiveness. Score close to 0 if co-founders are not explained nor mentioned.)
- Has the team performed any sort of market validation (pre-sell, survey, concierge, or other techniques such as wizard-of-oz)?(Score 0-10, with 10 being the most convincing demonstration or assertion of market validation performed).
- Stage in the prototyping or MVP phase (Score 1-10, with 10 closest to completeness of MVP.)

Ex. Business Model (5/10), Differentiator (5/10), ...

Calibrate the scores based on personality. For example, a harsh judge will give lower scores (closer to 0) than usual.
Clearly state the individual scores after each item and explain why.
Keep track of a cumulative score. The final score based on the average of the individual scores in the end.

Answer the following questions but state and summarize all in paragraph form:
- Is the business described B2B or B2C? What are the challenges related to this?
- Is it clear what problem is being solved, and who has the problem? Are these problems well-known by those in the industry?
- What industry or vertical does the business fall under? Is the space crowded? What other companies are known to address this problem?
- Are there any business model patterns the founding team needs to study more carefully?
- Has the team demonstrated compelling market validation?

Explain areas for improvement. Do you have tips on improving the business model? How would you rewrite the pitch to make it score at least an 8/10?

Do not make things up. Clearly state that information is lacking if not specified.

If things are not clear, ask!

Q1: <question 1>
...

Evaluate their pitch:
{pitch}
""".strip()
panelist_prompt_template = PromptTemplate(
    input_variables=['personality', 'pitch'],
    template=panelist_prompt_template_text,
)
personality_1 = """
- Your personality is harsh, brutal, frank, and blunt.
- You have decades of experience in industry.
- You are harsh, blunt, and abrasive at times, but fair.
- You dislike claims that are not substantiated.
- You dislike stylish pitches that are not accompanied by substance.
- You focus on weaknesses in the pitch.
- You do not suggest improvements.
- You tend to give low scores.
""".strip()
personality_2 = """
- Your personality is neutral but inquisitive.
- If a claim is not substantiated, you ask them to consider how they might find evidence for their claim.
- You focus on substance and ignore style.
- If you find weaknesses in the pitch, you ask them to expound on their thought process.
- You genuinely want to help the startup team.
""".strip()
personality_3 = """
- Your personality is optimistic.
- You have only a few years of experience in industry.
- You are optimistic about the merits of ideas.
- You focus on opportunities available to the group.
- You acknowledge weaknesses in the pitch, but you suggest improvements.
- You offer to help even outside the pitch.
- You tend to give high scores.
""".strip()
### Prefect
prefect_db = PGVector(
    embedding_function=embeddings,
    collection_name=config.PREFECT_LANGCHAIN_VECTOR_COLLECTION_NAME,
    connection_string=secret_config.PREFECT_PGVECTOR_CONNECTION_STRING,
)
prefect_retriever = prefect_db.as_retriever()
prefect_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=prefect_retriever,
)
## FastAPI
app = FastAPI()

def send_message_panelist(
    telegram_id: str,
    message: str
):
    if len(message) > 4096:
        part_length = 4096
        parts = [message[i: i + part_length] for i in range(0, len(message), part_length)]
        for part in parts:
            res = requests.post(
                f'{secret_config.PANELIST_TELEGRAM_BOT_ENDPOINT_BASE}/sendMessage',
                json={
                    'chat_id': telegram_id,
                    'text': part,
                }
            )
            print(res.text)
        return
    res = requests.post(
        f'{secret_config.PANELIST_TELEGRAM_BOT_ENDPOINT_BASE}/sendMessage',
        json={
            'chat_id': telegram_id,
            'text': message,
        },
    )
    return

def send_message_prefect(
    telegram_id: str,
    message: str
):
    if len(message) > 4096:
        part_length = 4096
        parts = [message[i: i + part_length] for i in range(0, len(message), part_length)]
        for part in parts:
            res = requests.post(
                f'{secret_config.PREFECT_TELEGRAM_BOT_ENDPOINT_BASE}/sendMessage',
                json={
                    'chat_id': telegram_id,
                    'text': part,
                }
            )
            print(res.text)
        return
    res = requests.post(
        f'{secret_config.PREFECT_TELEGRAM_BOT_ENDPOINT_BASE}/sendMessage',
        json={
            'chat_id': telegram_id,
            'text': message,
        },
    )
    return

@app.post('/webhook/panelist')
async def webhook_panelist(
    request: Request,
    response: Response,
):
    headers = request.headers
    body = await request.json()
    # print(headers)
    # print(body)
    # ards
    ## Guard: Update ID exists in cache?
    update_id = body['update_id']
    update_id_key = f'panelist_update_id:{update_id}'
    update_id_existed = redis_connection.set(
        update_id_key,
        1,
        config.UPDATE_ID_CACHE_DURATION_SECONDS,
        get=True,
    )
    if update_id_existed:
        logging.warning('Update ID already exists.')
        return
    ## Guard: has message?
    has_message = not not body.get('message')
    if not has_message:
        logging.warning('Has no message.')
        return
    ## Guard: is it in a private chat?
    is_private = body['message']['chat']['type'] == 'private'
    if not is_private:
        logging.warning('Not a private message.')
        return
    ## Guard: does the request have the secret header?
    secret_matches = headers.get('x-telegram-bot-api-secret-token') == secret_config.PANELIST_TELEGRAM_BOT_SECRET
    if not secret_matches:
        logging.warning(f'Secret does not match.')
        return
    # Webhook proper
    telegram_id = body['message']['from']['id']
    message_text = body['message']['text']
    if message_text.strip() == '/start':
        send_message_panelist(telegram_id, "Welcome to the ITE start-up training center. Send your pitch here to get it evaluated by our AI panelists.")
        return
    try:
        send_message_panelist(telegram_id, "Thank you for your pitch. We have three panelists with us today. Please give them some time to review your pitch.")
        send_message_panelist(telegram_id, f"""Here is Panelist 1's feedback.\n\n{panelist_qa.run(panelist_prompt_template.format(pitch=message_text, personality=personality_1))}""")
        send_message_panelist(telegram_id, f"""Here is Panelist 2's feedback.\n\n{panelist_qa.run(panelist_prompt_template.format(pitch=message_text, personality=personality_2))}""")
        send_message_panelist(telegram_id, f"""Here is Panelist 3's feedback.\n\n{panelist_qa.run(panelist_prompt_template.format(pitch=message_text, personality=personality_3))}""")
        send_message_panelist(telegram_id, "All panelists have responded. We hope their feedback will help you gain insight into your pitch.")
        return
    except Exception as e:
        logging.error(e)
        send_message_panelist(telegram_id, 'Sorry! We encountered an error. Please let the administrators know.')
        return

@app.post('/webhook/prefect')
async def webhook_prefect(
    request: Request,
    response: Response,
):
    headers = request.headers
    body = await request.json()
    # print(headers)
    # print(body)
    # ards
    ## Guard: Update ID exists in cache?
    update_id = body['update_id']
    update_id_key = f'prefect_update_id:{update_id}'
    update_id_existed = redis_connection.set(
        update_id_key,
        1,
        config.UPDATE_ID_CACHE_DURATION_SECONDS,
        get=True,
    )
    if update_id_existed:
        logging.warning('Update ID already exists.')
        return
    ## Guard: has message?
    has_message = not not body.get('message')
    if not has_message:
        logging.warning('Has no message.')
        return
    ## Guard: is it in a private chat?
    is_private = body['message']['chat']['type'] == 'private'
    if not is_private:
        logging.warning('Not a private message.')
        return
    ## Guard: does the request have the secret header?
    secret_matches = headers.get('x-telegram-bot-api-secret-token') == secret_config.PREFECT_TELEGRAM_BOT_SECRET
    if not secret_matches:
        logging.warning(f'Secret does not match.')
        return
    # Webhook proper
    telegram_id = body['message']['from']['id']
    message_text = body['message']['text']
    if message_text.strip() == '/start':
        send_message_prefect(telegram_id, "Hello! I can answer your questions about the Ateneo de Manila Unviersity Student Handbook.")
        return
    try:
        send_message_prefect(telegram_id, prefect_qa.run(message_text))
    except Exception as e:
        logging.error(e)
        send_message_prefect(telegram_id, 'Sorry! We encountered an error. Please let the administrators know.')
        return
