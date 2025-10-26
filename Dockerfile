FROM python:3.12
WORKDIR /app

RUN pip install pipenv

COPY Pipfile Pipfile.lock ./
RUN pipenv install --deploy --ignore-pipfile --system

COPY fitness_assistant /app/fitness_assistant
COPY data/fitness_exercises_500.csv data/fitness_exercises_500.csv

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "fitness_assistant.app:app"]

