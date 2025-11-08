FROM sweep-base:latest

WORKDIR /workspace/absolute-bert

COPY . .

RUN poetry install --no-dev --no-cache --extras sweep
RUN rm -rf /root/.cache/pip && rm -rf /root/.cache/pypoetry

RUN chmod a+x scripts/sweep_run-entrypoint.sh && \
    chmod a+x scripts/sweep_run-pull_and_run.sh

CMD ["scripts/sweep_run-entrypoint.sh"]