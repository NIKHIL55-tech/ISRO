# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="MOSDAC AI HelpBot")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def root():
#     return {"message": "MOSDAC API is running!"}

# @app.post("/query")
# def query_handler(query: dict):
#     user_query = query.get("query", "")
#     # TODO: Later integrate KG or RAG pipeline here
#     return {"response": f"You asked: {user_query}. This is a placeholder response."}


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.vector_search import vector_search
# from src.kg_search import kg_query  # Optional: use if Member A's KG is ready

app = FastAPI()

# Allow frontend to call the backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "MOSDAC API is running!"}

@app.post("/vector-query")
async def vector_query_handler(request: Request):
    data = await request.json()
    user_query = data.get("query", "")
    response = vector_search(user_query)
    return {"response": response}

# Optional: enable when KG is ready
# @app.post("/kg-query")
# async def kg_query_handler(request: Request):
#     data = await request.json()
#     user_query = data.get("query", "")
#     response = kg_query(user_query)
#     return {"response": response}
