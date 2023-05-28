FROM tiangolo/uwsgi-nginx-flask:python3.9

COPY ./app /app

EXPOSE 8000

RUN pip install -r requirements.txt

CMD ["python", "app.py"]