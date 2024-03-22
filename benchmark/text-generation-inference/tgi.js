import { get_options, run } from "./common.js";

const host = __ENV.HOST || '127.0.0.1:8080';
const vu = __ENV.VU || 1;
const max_new_tokens = 250;

function generate_payload(prompt){
    return {
        "inputs": prompt,
        "parameters": {
            "do_sample": false,
            "max_new_tokens": max_new_tokens,
            "decoder_input_details": true
        }
    }
}

export const options = get_options();

export default function(){
    run(host, vu, generate_payload, max_new_tokens);
}

export function handleSummary(data) {
    let filename = (new Date()).toISOString() + '_tgi_VU_' + vu + '_summary.json'
    let result = {}
    result[filename] = JSON.stringify(data, null, 4)
    return result
}
