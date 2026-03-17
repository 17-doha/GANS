FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

COPY . .

# Updated the path to point directly to the main.py file inside the app folder
CMD ["conda", "run", "-n", "mlops_assignmemt", "python", "-m", "app.main"]