FROM sweep-base:latest

WORKDIR /workspace

COPY . absolute-bert/

WORKDIR /workspace/absolute-bert

RUN poetry install --extras gpu --extras benchmark

CMD ["scripts/sweep_run-entrypoint.sh"]