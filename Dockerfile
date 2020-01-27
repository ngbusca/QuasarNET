FROM python:3.7-buster

WORKDIR /services/qnet

RUN apt update && apt install -y git-all && git clone https://github.com/ngbusca/QuasarNET.git && cd QuasarNET && pip install -r api/requirements.txt && pip install -r requirements.txt && python setup.py install

EXPOSE 5000

CMD ["python", "QuasarNET/api/api.py"]
