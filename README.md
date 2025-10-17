# Fitness Assistant

Staying consistent with fitness routines is challenging, especially for beginners. Gyms can be intimidating, and personal trainers aren't always available.

The Fitness Assistant provides a conversational AI that helps users choose exercises and find alternatives, making fitness more manageable.

# Dataset
The dataset used in this project contains information about various exercises, including:

- Exercise Name: The name of the exercise (e.g., Push-Ups, Squats).
- Type of Activity: The general category of the exercise (e.g., Strength, Mobility, Cardio).
- Type of Equipment: The equipment needed for the exercise (e.g., Bodyweight, Dumbbells, Kettlebell).
- Body Part: The part of the body primarily targeted by the exercise (e.g., Upper Body, Core, Lower Body).
- Type: The movement type (e.g., Push, Pull, Hold, Stretch).
- Muscle Groups Activated: The specific muscles engaged during the exercise (e.g., Pectorals, Triceps, Quadriceps).
- Instructions: Step-by-step guidance on how to perform the exercise correctly.
The dataset was generated using ChatGPT and contains 500 records. It serves as the foundation for the Fitness Assistant's exercise recommendations and instructional support.

You can find the data in data/data.csv.

## Technologies

* Elasticsearch for hybrid search
* OpenAI as LLM
* Flask as the API inference

## Running it

### Installing dependencies

We use pipenv to manage dependencies and Python 3.12

Make sure you have pipenv installed:

```bash
pip install pipenv
```

Installing the dependencies:

```bash
pipenv install
```
### Running the application

Running the Flask application:
```bash
pipenv run python -m fitness_assistant.app
```
Testing it:
```
curl -X POST http://localhost:5000/question \
     -H 'Content-Type: application/json' \
     -d '{"question": "What equipment do I need to perform the Push-Up Hold exercise?"}' 
```

Sending feedback:
```
curl -X POST http://localhost:5000/feedback \
     -H 'Content-Type: application/json' \
     -d '{"conversation_id": "d8bde824-8a7f-40ff-947d-df7f67032ec5", "feedback": 1}'
```

### Misc

Running Jupyter notebook for experiments:
```bash
cd notebooks
pipenv run jupyter notebook
```
## Interface

We use Flask for serving the application as API.

## Ingestion

The ingestion script is in [fitness_assistant/ingest.py](fitness_assistant/ingest.py)

## Evaluation

The basic approach using Elasticsearch with default boosting - gave the following results:

* hit_rate: 70%
* MRR: 59%

After tuning the parameters, the result is:

* hit_rate: 68%
* MRR: 62%

The best parameters are:
{
'exercise_name': 2.5,
'type_of_activity': 1.4,
'type_of_equipment': 2.2,
'body_part': 0.5,
'type': 2.5,
'muscle_groups_activated': 0.06,
'instruction': 0.5
}

### Retrieval

### RAG Flow

We use LLM-as-a-Judge to evaluate our RAG flow.

Among 200 records:
* Using GPT-4o-mini:
    * 67% RELEVANT
    * 14% PARTLY_RELEVANT
    * 19% IRRELEVANT

* Using GPT-4.1:
    * 77.5% RELEVANT
    * 14% PARTLY_RELEVANT
    * 8.5% IRRELEVANT

## Monitoring

