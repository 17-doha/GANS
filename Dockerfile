FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

COPY . .

CMD ["conda", "run", "-n", "mlops_assignmemt", "python", "main.py"]