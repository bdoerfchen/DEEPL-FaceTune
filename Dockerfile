FROM python:3.11

WORKDIR /app

# Install necessary pip modules
RUN pip install \
            eel \
            opencv-python-headless \
            pillow \
            numpy \
            tensorflow==2.15.0 \
            keras==2.15.0 
# Copy internal source files
COPY src/ src/

# Container is healthy if eel/bottle serves index.html
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 CMD [ "curl", "localhost:8080/" ]

EXPOSE 8080
# Run main.py as unprivileged user
USER 1000
ENTRYPOINT [ "python3.11", "-u", "src/main.py" ]
