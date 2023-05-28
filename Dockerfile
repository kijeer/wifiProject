<<<<<<< HEAD
FROM tiangolo/uwsgi-nginx-flask:python3.9

WORKDIR /app

COPY ./app /app/

EXPOSE 80

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
=======
FROM tiangolo/uwsgi-nginx-flask:python3.9

COPY ./app /app

EXPOSE 80
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
>>>>>>> e9b23a010084673d79b90c3db24313e1ca4a5163
