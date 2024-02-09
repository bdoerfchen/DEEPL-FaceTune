FROM python:3.11

WORKDIR /app
RUN pip install \
            eel \
            opencv-python-headless \
            pillow \
            numpy \
            tensorflow==2.15.0 \
            keras==2.15.0 
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 CMD [ "curl", "localhost:8000/" ]
USER 1000
ENTRYPOINT [ "python3.11", "-u", "src/main.py" ]
