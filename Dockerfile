FROM python:3.11

WORKDIR /app
#opencv wrong?
RUN pip install eel opencv-python numpy tensorflow==2.15.0 keras==2.15.0 
COPY . .

# TODO: Add non-root user
EXPOSE 8000
ENTRYPOINT [ "python3.11", "src/main.py" ]
