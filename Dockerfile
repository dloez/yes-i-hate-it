FROM python:3.8

WORKDIR /usr/local/yes-i-hate-it
COPY . .

RUN python -m pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install --system
RUN pip install .

CMD ['python', 'src/yes_i_hate_it/main.py']
