import pprint

import requests
from prometheus_client.parser import text_string_to_metric_families


def get_node_results(node_url):

    metrics = requests.get(node_url + "/metrics").text

    counters = {
        "tgi_request_input_length": {},
        "tgi_request_generated_tokens": {},
        "tgi_request_mean_time_per_token_duration": {},
        "tgi_batch_inference_duration": {},
        "tgi_request_queue_duration": {},
    }

    for family in text_string_to_metric_families(metrics):
        if family.name in counters:
            for sample in family.samples:
                if sample.name == family.name + "_sum":
                    if len(sample.labels) == 0:
                        counters[family.name]["sum"] = sample.value
                    elif "method" in sample.labels:
                        counters[family.name][sample.labels["method"] + "_sum"] = sample.value
                elif sample.name == family.name + "_count":
                    if len(sample.labels) == 0:
                        counters[family.name]["count"] = sample.value
                    elif "method" in sample.labels:
                        counters[family.name][sample.labels["method"] + "_count"] = sample.value

    num_requests = counters["tgi_request_mean_time_per_token_duration"]["count"]
    input_tokens = counters["tgi_request_input_length"]["sum"]
    avg_time_per_token = counters["tgi_request_mean_time_per_token_duration"]["sum"] * 1000 / num_requests
    prefill_time = counters["tgi_batch_inference_duration"]["prefill_sum"]
    decode_time = counters["tgi_batch_inference_duration"]["decode_sum"]
    total_time = prefill_time + decode_time
    decode_tokens = counters["tgi_request_generated_tokens"]["sum"]
    avg_queue_duration = counters["tgi_request_queue_duration"]["sum"] / num_requests

    return {
        "requests": num_requests,
        "mean_input_tokens": input_tokens / num_requests,
        "avg_time_per_token": avg_time_per_token,
        "throughput": (input_tokens + decode_tokens) / total_time,
        "prefill_throughput": input_tokens / prefill_time,
        "decode_throughput": decode_tokens / decode_time,
        "time_to_first_token": avg_queue_duration + (prefill_time / num_requests),
    }


for port in [8081, 8082, 8083]:
    pprint.pprint(get_node_results(f"http://0.0.0.0:{port}"))
