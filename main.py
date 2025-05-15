from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid

from kdtree_audio_index import KDTreeAudioIndexer

app = FastAPI()

UPLOAD_FOLDER = "static/uploads"
DATA_FOLDER ="static/data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

indexer = KDTreeAudioIndexer()
indexer.load_data_and_build_tree(DATA_FOLDER)

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": None,
        "uploaded_file": None,
    })

@app.post("/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile):
    ext = os.path.splitext(file.filename)[-1]
    filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    results = indexer.query(path, k=3)
    indexer.index_audio_file(path, filename)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "uploaded_file": filename,
    })
