import uvicorn

from app import app

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=5000, workers=2)
    #for production see more https://www.uvicorn.org/deployment/

    # for development uvicorn.run("main:app", host='0.0.0.0', port=5000, reload=True)
