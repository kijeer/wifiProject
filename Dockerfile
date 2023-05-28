FROM tiangolo/uwsgi-nginx-flask:python3.9

COPY ./app /app

EXPOSE 80
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
