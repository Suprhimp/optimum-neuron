# NeuronX TGI benchmark using mulitple replicas

## Prerequisites

- install [k6](https://k6.io/docs/get-started/installation/)
- install [prometheus_client](https://github.com/prometheus/client_python)

## Select model and configuration

Edit the `.env` file to select the model to use for the benchmark and its configuration.

## Start the servers

```shell
docker compose --env-file llama-7b/.env up
```

Note: replace the .env file to change the model configuration

## Run the benchmark

```shell
$ k6 run tgi.js -e VU=96 -e DURATION='300s'
```

Where VU is the number of virtual users and DURATION the duration of the run.

By default, the load test runs with the same input. To use random inputs, set RANDOM_INPUT to true:

```shell
$ k6 run tgi.js -e VU=96 -e RANDOM_INPUT=true
```

## Query the server metrics

This will fetch the results from one replica only, but they should be equivalents.

```shell
$ python tgi_metrics.py
```

## Generate aggregated results summary

```
$ python -m pip install panda requests
$ python generate_csv.py
```
