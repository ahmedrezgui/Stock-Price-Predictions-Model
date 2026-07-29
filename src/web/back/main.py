from fastapi import FastAPI, HTTPException
from fastapi.params import Body
from fastapi.middleware.cors import CORSMiddleware
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.path.dirname(__file__)) / '..' / '..' / '..'))
from src import CacheService
from src.services import (
    stock_data_service,
    CompanyNotFoundError,
    DataUnavailableError,
    ModelUnavailableError,
)
sys.path.insert(0, str(Path(os.path.dirname(__file__)) / '..' / '..'))
from rag import create_agents_graph
from dotenv import load_dotenv
import json


load_dotenv()

DISABLE_BACKEND_CACHE = os.environ.get("DISABLE_BACKEND_CACHE").lower() == "true" if os.environ.get("DISABLE_BACKEND_CACHE") != None else True
BACKEND_CACHE_CONNECTION_STRING = os.environ.get("BACKEND_CACHE_CONNECTION_STRING") if os.environ.get("BACKEND_CACHE_CONNECTION_STRING") != None else ""
BACKEND_CACHE_EXPIRATION_TIME = int(os.environ.get("BACKEND_CACHE_EXPIRATION_TIME")) if os.environ.get("BACKEND_CACHE_EXPIRATION_TIME") != None else 0

cacheService = CacheService.getInstance(BACKEND_CACHE_CONNECTION_STRING, DISABLE_BACKEND_CACHE, BACKEND_CACHE_EXPIRATION_TIME)
agents = create_agents_graph()

app = FastAPI()

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


@app.post("/rag")
def rag_query(query: str = Body(...)):
    """
    Endpoint to handle RAG queries.
    """
    try:
        input = {"question": query}
        response = agents.invoke(input, stream_mode="values")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
