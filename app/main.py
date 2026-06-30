from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api import router as api_router
from app.database import init_db
from config import settings

# Initialize Database
init_db()

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# Mount static files
app.mount("/static", StaticFiles(directory=str(settings.BASE_DIR / "static")), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(settings.BASE_DIR / "templates"))

# Include API router
app.include_router(api_router, prefix="/api")

@app.get("/")
async def serve_home(request: Request):
    """Serves the home page."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/dashboard")
async def serve_dashboard(request: Request):
    """Serves the main dashboard page."""
    return templates.TemplateResponse(request=request, name="dashboard.html")
