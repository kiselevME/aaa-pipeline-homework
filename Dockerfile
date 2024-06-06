FROM python:3.11-slim

RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

WORKDIR app

COPY ./requirements.txt $WORKDIR/
RUN pip3 install --no-cache-dir -r $WORKDIR/requirements.txt

COPY . $WORKDIR
ENTRYPOINT ["python", "main.py"]
# RUN PYTHONPATH="$WORKDIR:$PYTHONPATH"