import os
import time
from tqdm import tqdm
import numpy as np

from examples.infer_trt import get_engine_context, trt_inference


def policy_profiler(engine_path: str, infer_count: int) -> None:
    load_start_time = time.time()

    # Get trt engine and context
    engine, context = get_engine_context(engine_path=engine_path)

    infer_start_time = time.time()
    loading_time = infer_start_time - load_start_time
    print(f'Policy loading time: {loading_time:.2f} s')

    for i in tqdm(range(infer_count)):
        # Define random inputs
        obs = np.random.random((1, 256)).astype(np.float32)
        rf = np.array([[False]], dtype=bool)

        trt_outputs = trt_inference(engine=engine, context=context, inputs=[obs, rf])

        action = trt_outputs[-1]

    infer_end_time = time.time()
    total_infer_time = infer_end_time - infer_start_time
    avg_infer_time = total_infer_time / infer_count
    print(f'Total infer time: {total_infer_time:.4f} s, avg infer time: {avg_infer_time:.4f} s')


if __name__ == '__main__':
    # Define constants
    trt_model_path: str = 'models/policy.trt'
    trt_model_path = os.path.join(os.path.dirname(__file__), trt_model_path)

    # Define variables
    infer_count: int = 1000

    # Do policy infer profiler
    policy_profiler(engine_path=trt_model_path, infer_count=infer_count)

