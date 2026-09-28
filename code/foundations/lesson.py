"""Run with a topic name. All values are illustrative or derived, never benchmarks."""
import argparse
import json
import math
import struct

from model import Decoder, attention_row, linear, swiglu


def tensor_file():
    """A valid, in-memory safetensors file with a padded 120-byte JSON header."""
    descriptor = {"toy": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
    header = json.dumps(descriptor, separators=(",", ":")).encode()
    header += b" " * (120 - len(header))
    return struct.pack("<Q", len(header)) + header + struct.pack("<6f", *range(1, 7))


def inspect_tensor(blob):
    """Teaching subset: one contiguous F32 tensor; not a production format loader."""
    if len(blob) < 8:
        raise ValueError("missing header length")
    length, = struct.unpack_from("<Q", blob)
    if length > 1_000_000 or 8 + length > len(blob):
        raise ValueError("bad header length")
    header = json.loads(blob[8:8+length])
    if set(header) != {"toy"}:
        raise ValueError("expected one tensor")
    tensor = header["toy"]
    shape = tensor["shape"]
    if tensor["dtype"] != "F32" or shape != [2, 3]:
        raise ValueError("unsupported dtype or shape")
    begin, end = tensor["data_offsets"]
    if type(begin) is not int or type(end) is not int:
        raise ValueError("offsets must be integers")
    origin = 8 + length
    if begin != 0 or end - begin != math.prod(shape) * 4 or origin + end != len(blob):
        raise ValueError("bad payload interval")
    return {"origin": origin, "element_1_2": struct.unpack_from("<f", blob, origin+20)[0],
            "interval": [origin+begin, origin+end]}


def demonstrate(topic):
    if topic == "attention":
        p, y = attention_row([1., 0.], [[0., 0.], [math.sqrt(2)*math.log(3), 0.]],
                             [[2., 0.], [0., 4.]])
        return {"probabilities": p, "output": y}
    if topic == "ffn":
        x = [1., -1.]
        relu = [max(0, v) for v in linear(x, [[1., 0.], [0., 1.], [1., -1.]])]
        return {"relu_hidden": relu, "relu_update": linear(relu, [[1., 1., 0.], [0., 1., 1.]]),
                "swiglu_update": swiglu(x, [[1., 1.], [1., 0.]], [[2., 0.], [3., 0.]],
                                        [[1., -1.], [.5, 1.]])}
    if topic == "decoder":
        model = Decoder()
        prompt = [0, 1, 2]
        cache = model.cache()
        cached = [cache.append(t) for t in prompt]
        full = model.full(prompt)
        error = max(abs(a-b) for row, ref in zip(cached, full) for a, b in zip(row, ref))
        next_token = max(range(model.cfg.vocab), key=full[-1].__getitem__)
        before = cache.position
        cache.append(next_token)
        return {"prompt": prompt, "last_logits": full[-1], "maximum_error": error,
                "chosen_token": next_token, "evaluated_before_feedback": before,
                "evaluated_after_feedback": cache.position}
    if topic == "formats":
        return inspect_tensor(tensor_file())
    if topic == "numerics":
        values, scale = [-1.1, .2, 1.6], .5
        codes = [max(-7, min(7, round(x/scale))) for x in values]
        return {"codes": codes, "reconstruction": [q*scale for q in codes],
                "bytes_for_32_codes_plus_scale": (32*4+16)//8}
    if topic == "cpu":
        return {"complete_eight_lane_vectors_and_tail": divmod(19, 8),
                "sum_of_1_to_19": sum(range(1, 20)), "f32_per_64_byte_line": 64//4}
    if topic == "gpu":
        return {"blocks_at_24_KiB": 64//24, "blocks_at_16_KiB": 64//16,
                "transfer_ms": 64*2**20 / (16*2**30) * 1000}
    if topic == "units":
        bytes_per_token = 2*2*2*4*2
        return {"kv_bytes_per_token": bytes_per_token, "logical_bytes": 3*5*bytes_per_token,
                "reserved_bytes": 3*2*4*bytes_per_token, "lower_bound_ms": max(2e9/100e9,8e9/1e12)*1000}
    if topic == "measurement":
        send, first, last, terminal = 3, 17, 40, 44
        return {"ttft_ms": first-send, "gaps_ms": [25-first, last-25],
                "mean_gap_ms": (last-first)/2, "terminal_ms": terminal-send}
    raise ValueError(topic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", choices=["attention","ffn","decoder","formats","numerics",
                                         "cpu","gpu","units","measurement"])
    print(json.dumps(demonstrate(parser.parse_args().topic), indent=2))
