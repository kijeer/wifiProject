FROM tiangolo/uwsgi-nginx-flask:python3.9

WORKDIR /app

COPY ./app /app/

EXPOSE 80

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]