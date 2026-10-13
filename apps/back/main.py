from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import os
from legacy import CacheService
from market_data import (
    stock_data_service,
    CompanyNotFoundError,
    DataUnavailableError,
    ModelUnavailableError,
)
def ask(*args, **kwargs):
    from stock_assistant import ask as run
    global agents
    if agents is None:
        agents = create_agents_graph()
    return run(agents, *args[1:], **kwargs)

def create_agents_graph():
    from stock_assistant import create_agents_graph as create
    return create()

from dotenv import load_dotenv
import json
import uuid


load_dotenv()

DISABLE_BACKEND_CACHE = os.environ.get("DISABLE_BACKEND_CACHE").lower() == "true" if os.environ.get("DISABLE_BACKEND_CACHE") != None else True
BACKEND_CACHE_CONNECTION_STRING = os.environ.get("BACKEND_CACHE_CONNECTION_STRING") if os.environ.get("BACKEND_CACHE_CONNECTION_STRING") != None else ""
BACKEND_CACHE_EXPIRATION_TIME = int(os.environ.get("BACKEND_CACHE_EXPIRATION_TIME")) if os.environ.get("BACKEND_CACHE_EXPIRATION_TIME") != None else 0

cacheService = CacheService.getInstance(BACKEND_CACHE_CONNECTION_STRING, DISABLE_BACKEND_CACHE, BACKEND_CACHE_EXPIRATION_TIME)
agents = None

app = FastAPI()
from stock_forecasting.api import create_app as create_forecast_app
app.mount("/forecast", create_forecast_app())

# CORS config for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/companies")
def list_companies():
    key = "companies"
    if (DISABLE_BACKEND_CACHE == False and cacheService != None and cacheService.exist(key) == True):
        return json.loads(cacheService.get(key))
    try:
        res = stock_data_service.list_companies()
        if (DISABLE_BACKEND_CACHE == False and cacheService != None):
            cacheService.set(key, json.dumps(res))
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock/{company}")
def get_stock(company: str):
    key = f"stock/{company}"
    if (DISABLE_BACKEND_CACHE == False and cacheService != None and cacheService.exist(key) == True):
        return json.loads(cacheService.get(key))

    try:
        res = stock_data_service.get_stock_with_forecast(company)
    except CompanyNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelUnavailableError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except DataUnavailableError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

    if (DISABLE_BACKEND_CACHE == False and cacheService != None):
        cacheService.set(key, json.dumps(res))
    return res


def parse_chat_request(raw_body: str):
    """
    Read a {"question", "thread_id"} body.

    Clients that still POST the bare question as plain text keep working; each of
    those gets a fresh conversation, since they carry no thread of their own.
    """
    try:
        payload = json.loads(raw_body)
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        question = payload.get("question") or ""
        thread_id = payload.get("thread_id") or str(uuid.uuid4())
    else:
        question = payload if isinstance(payload, str) else raw_body
        thread_id = str(uuid.uuid4())

    return question.strip(), thread_id


@app.post("/chat")
async def chat_query(request: Request):
    """
    Endpoint to handle assistant chat queries.
    """
    raw_body = (await request.body()).decode("utf-8").strip()
    question, thread_id = parse_chat_request(raw_body)
    if not question:
        raise HTTPException(status_code=400, detail="A question is required")

    try:
        answer = await run_in_threadpool(ask, agents, question, thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # answer carries the text, its sources, and any charts drawn for it.
    return {**answer, "thread_id": thread_id}

