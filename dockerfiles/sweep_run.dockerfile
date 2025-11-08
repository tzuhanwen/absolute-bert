FROM sweep-base:latest

WORKDIR /workspace

COPY . absolute-bert/

WORKDIR /workspace/absolute-bert

RUN poetry install --extras gpu --extras benchmark

RUN chmod a+x scripts/sweep_run-entrypoint.sh && \
    chmod a+x scripts/sweep_run-pull_and_run.sh

CMD ["scripts/sweep_run-entrypoint.sh"]