ARG RUNTIME_VERSION="3.8"
ARG DISTRO_VERSION="buster"

FROM python:${RUNTIME_VERSION}-slim-${DISTRO_VERSION}

MAINTAINER Mokarrom Hossain <mokarrom@live.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         curl \
         ca-certificates \
         libffi6 \
         libffi-dev \
         libgomp1\
         python3-pip \
         python3-setuptools \
         zip \
         unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip3 --no-cache-dir install gunicorn flask openai python-dotenv boto3

COPY requirements.txt .
COPY src/summarizer /opt/program/summarizer
COPY src/sm_container /opt/program

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chmod +x /opt/program/serve

WORKDIR /opt/program
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"



