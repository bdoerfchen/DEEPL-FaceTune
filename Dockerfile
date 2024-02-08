FROM ubuntu:22.04
USER root

# Install
RUN apt update && \
    apt install -y python3.11 python3-pip python3.11-venv

WORKDIR /app
#Install directly instead of requirements.txt
RUN python3.11 -m venv .venv && \
    . .venv/bin/activate && \
    pip install eel opencv-python numpy tensorflow==2.15.0 keras==2.15.0 
COPY . .

# TODO: Add non-root user
EXPOSE 8000
ENTRYPOINT [ "python3.11", "src/main.py" ]
# FIX: entrypoint doesnt use venv
