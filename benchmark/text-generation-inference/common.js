import { check, randomSeed } from 'k6';
import http from 'k6/http';
import { SharedArray } from 'k6/data';
import { Trend, Counter } from 'k6/metrics';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

const seed = 0;

const host = __ENV.HOST || '127.0.0.1:8080';
const duration = __ENV.DURATION || '300s';
const vu = __ENV.VU || 1;
const random_input = __ENV.RANDOM_INPUT || false;
const timePerToken = new Trend('time_per_token', true);
const tokens = new Counter('tokens');
const new_tokens = new Counter('new_tokens');
const input_tokens = new Counter('input_tokens');

randomSeed(seed);
const samples = new SharedArray('ShareGPT samples', function () {
    return JSON.parse(open('./samples.json'));
});

export function get_options(){
    return {
        thresholds: {
            http_req_failed: ['rate<0.1'],
        },
        scenarios: {
            load_test: {
                executor: 'constant-vus',
                duration: duration,
                vus: vu,
            },
        },
    };
}


export function run(host, vu, generate_payload, max_new_tokens) {
    const headers = {'Content-Type': 'application/json'};
    let prompt = ''
    if (random_input) {
        prompt = randomItem(samples);
    } else {
        prompt = samples[0];
    }
    const payload = JSON.stringify(generate_payload(prompt));
    const res = http.post(`http://${host}/generate`, payload, {
        headers,
    });
    if (res.status >= 400 && res.status < 500) {
        return;
    }

    check(res, {
        'Post status is 200': (r) => res.status === 200,
    });
    const duration = res.timings.duration;

    if (res.status === 200) {
        const body = res.json();
        const n_tokens = body.details.tokens.length;
        const latency_ms_per_token = duration / n_tokens;
        timePerToken.add(latency_ms_per_token);
        const latency_in_s = latency_ms_per_token / 1000;
        const individual_throughput = 1 / latency_in_s;
        const _input_tokens = body.details.prefill.length;
        tokens.add(n_tokens + _input_tokens);
        input_tokens.add(_input_tokens);
        new_tokens.add(n_tokens);
    }
}
