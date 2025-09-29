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

# Running it

We use pipenv to manage dependencies and Python 3.12

Make sure you have pipenv installed:

```bash
pip install pipenv
```

Installing the dependencies:

```bash
pipenv install
```

Running Jupyter notebook for experiments:
```
cd notebooks
pipenv run jupyter notebook
```
## Evaluation

The basic approach using Minsearch without using any boosting - gave the following results:

* hit_rate: 63%
* MRR: 58%

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

## Monitoring

## Ingestion