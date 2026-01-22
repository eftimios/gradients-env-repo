import docker
import time
import requests
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_NAME = "iamPi/environment_test"
VLLM_IMAGE = "vllm/vllm-openai:latest"
AGENTGYM_IMAGE = "affinefoundation/agentgym:alfworld"
NETWORK_NAME = "agent_eval_net"

# Evaluation Params
NUM_EVALS = 2500  # All samples
DATA_LEN_RANGE = 2500
TEMPERATURE = 0.0
RANDOM_SEED = 42
NUM_WORKERS = 2  # Number of parallel workers (2 GPUs)

client = docker.from_env()
results_lock = Lock()
all_results = []


def evaluate_single_task(worker_id, task_id, vllm_port, agentgym_port, inference_model_name):
    """Evaluate a single task on a specific worker."""
    try:
        payload = {
            "model": inference_model_name,
            "base_url": f"http://vllm-server-{worker_id}:8000/v1",
            "task_id": task_id,
            "temperature": TEMPERATURE,
            "max_round": 30
        }

        start_ts = time.time()
        response = requests.post(f"http://localhost:{agentgym_port}/evaluate", json=payload, timeout=2500)
        result = response.json()

        latency = result.get('time_taken', time.time() - start_ts)
        score = result.get('score', 0.0)

        result_data = {
            "task_id": task_id,
            "task_name": result.get('task_name', 'unknown'),
            "score": score,
            "success": result.get('success', False),
            "time": latency,
            "error": result.get('error'),
            "worker_id": worker_id
        }

        with results_lock:
            all_results.append(result_data)

        return f"✓ Worker {worker_id} | Task {task_id} | Score: {score}"

    except Exception as e:
        error_msg = f"✗ Worker {worker_id} | Task {task_id} | Error: {e}"
        with results_lock:
            all_results.append({
                "task_id": task_id,
                "task_name": "error",
                "score": 0.0,
                "success": False,
                "time": 0.0,
                "error": str(e),
                "worker_id": worker_id
            })
        return error_msg


def run_parallel_eval_suite():
    containers = {}

    try:
        # 1. Infrastructure Setup
        print("🔧 Setting up network...")
        networks = client.networks.list(names=[NETWORK_NAME])
        if not networks:
            client.networks.create(NETWORK_NAME, driver="bridge")

        # Determine model name for inference
        if LORA_MODEL_NAME:
            inference_model_name = "trained_lora"
        else:
            inference_model_name = BASE_MODEL_NAME

        # 2. Start vLLM and AgentGym servers for each GPU
        for worker_id in range(NUM_WORKERS):
            gpu_id = worker_id
            vllm_port = 8000 + worker_id * 10
            agentgym_port = 8001 + worker_id * 10

            print(f"\n🚀 Starting Worker {worker_id} (GPU {gpu_id}):")

            # Start vLLM server
            if LORA_MODEL_NAME:
                print(f"  └─ vLLM: {BASE_MODEL_NAME} w/ LoRA {LORA_MODEL_NAME}")
                vllm_command = f"--model {BASE_MODEL_NAME} --enable-lora --lora-modules trained_lora={LORA_MODEL_NAME} --gpu-memory-utilization 0.9 --port 8000 --trust-remote-code"
            else:
                print(f"  └─ vLLM: {BASE_MODEL_NAME}")
                vllm_command = f"--model {BASE_MODEL_NAME} --gpu-memory-utilization 0.9 --port 8000 --trust-remote-code"

            vllm = client.containers.run(
                VLLM_IMAGE,
                command=vllm_command,
                name=f"vllm-server-{worker_id}",
                detach=True,
                network=NETWORK_NAME,
                ports={'8000/tcp': vllm_port},
                device_requests=[docker.types.DeviceRequest(device_ids=[str(gpu_id)], capabilities=[['gpu']])],
            )
            containers[f'vllm_{worker_id}'] = vllm

            # Start AgentGym server
            print(f"  └─ AgentGym server")
            agent = client.containers.run(
                AGENTGYM_IMAGE,
                name=f"agentgym-server-{worker_id}",
                detach=True,
                network=NETWORK_NAME,
                ports={'8000/tcp': agentgym_port}
            )
            containers[f'agent_{worker_id}'] = agent

        # 3. Wait for all vLLM servers to be ready
        print("\n⏳ Waiting for all vLLM servers to be ready...")
        for worker_id in range(NUM_WORKERS):
            vllm_port = 8000 + worker_id * 10
            while True:
                try:
                    if requests.get(f"http://localhost:{vllm_port}/v1/models", timeout=2).status_code == 200:
                        print(f"  ✅ Worker {worker_id} ready")
                        break
                except:
                    time.sleep(5)

        # 4. Generate evaluation task list
        random.seed(RANDOM_SEED)
        eval_list = list(range(1, DATA_LEN_RANGE + 1))  # All 2500 tasks
        random.shuffle(eval_list)
        eval_list = eval_list[:NUM_EVALS]  # Take first NUM_EVALS

        print(f"\n🎯 Starting parallel evaluation of {NUM_EVALS} tasks with {NUM_WORKERS} workers...")
        print(f"📊 Expected speedup: ~{NUM_WORKERS}x\n")

        # 5. Parallel Evaluation with ThreadPoolExecutor
        start_time = time.time()
        completed_count = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks, distributing them round-robin across workers
            futures = []
            for i, task_id in enumerate(eval_list):
                worker_id = i % NUM_WORKERS
                vllm_port = 8000 + worker_id * 10
                agentgym_port = 8001 + worker_id * 10

                future = executor.submit(
                    evaluate_single_task,
                    worker_id,
                    task_id,
                    vllm_port,
                    agentgym_port,
                    inference_model_name
                )
                futures.append(future)

            # Process results as they complete
            for future in as_completed(futures):
                result_msg = future.result()
                completed_count += 1
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                eta = (NUM_EVALS - completed_count) / rate if rate > 0 else 0

                print(f"[{completed_count}/{NUM_EVALS}] {result_msg} | Rate: {rate:.2f}/s | ETA: {eta/60:.1f}m")

        # 6. Calculate final statistics
        total_time = time.time() - start_time
        total_score = sum(r['score'] for r in all_results)
        avg_score = total_score / len(all_results) if all_results else 0
        avg_episode_time = sum(r['time'] for r in all_results) / len(all_results) if all_results else 0
        successful = sum(1 for r in all_results if r['success'])

        # 7. Save results to file
        safe_model_name = BASE_MODEL_NAME.split("/")[1]
        if LORA_MODEL_NAME:
            safe_lora_name = LORA_MODEL_NAME.split("/")[1]
            filename = f"eval_results_{safe_model_name}_{safe_lora_name}_parallel.txt"
        else:
            filename = f"eval_results_{safe_model_name}_parallel.txt"

        with open(filename, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"PARALLEL EVALUATION REPORT - {datetime.now()}\n")
            f.write(f"Model: {BASE_MODEL_NAME}\n")
            if LORA_MODEL_NAME:
                f.write(f"LoRA: {LORA_MODEL_NAME}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"CONFIGURATION:\n")
            f.write(f"- Workers: {NUM_WORKERS}\n")
            f.write(f"- Temperature: {TEMPERATURE}\n")
            f.write(f"- Max Rounds: 30\n\n")

            f.write(f"SUMMARY STATS:\n")
            f.write(f"- Total Tasks: {len(all_results)}\n")
            f.write(f"- Successful: {successful}\n")
            f.write(f"- Failed: {len(all_results) - successful}\n")
            f.write(f"- Average Score: {avg_score:.4f}\n")
            f.write(f"- Success Rate: {successful/len(all_results)*100:.2f}%\n")
            f.write(f"- Total Evaluation Time: {total_time/60:.2f} minutes\n")
            f.write(f"- Average Episode Time: {avg_episode_time:.2f}s\n")
            f.write(f"- Throughput: {len(all_results)/total_time:.2f} tasks/second\n\n")

            f.write("DETAILED RESULTS:\n")
            f.write(f"{'Task ID':<10} | {'Worker':<8} | {'Name':<15} | {'Score':<7} | {'Success':<8} | {'Time':<7}\n")
            f.write("-" * 80 + "\n")

            for res in sorted(all_results, key=lambda x: x['task_id']):
                f.write(f"{res['task_id']:<10} | {res['worker_id']:<8} | {res['task_name']:<15} | "
                       f"{res['score']:<7.2f} | {str(res['success']):<8} | {res['time']:<7.2f}s\n")
                if res['error']:
                    f.write(f"   └─ Error: {res['error']}\n")

        print(f"\n{'='*80}")
        print(f"✅ EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"📊 Results:")
        print(f"   - Tasks Completed: {len(all_results)}")
        print(f"   - Success Rate: {successful/len(all_results)*100:.2f}%")
        print(f"   - Average Score: {avg_score:.4f}")
        print(f"   - Total Time: {total_time/60:.2f} minutes")
        print(f"   - Throughput: {len(all_results)/total_time:.2f} tasks/second")
        print(f"\n💾 Results saved to: {filename}")
        print(f"{'='*80}\n")

    finally:
        print("\n🧹 Cleaning up containers...")
        for name, container in containers.items():
            try:
                print(f"  └─ Removing {name}")
                container.remove(force=True)
            except Exception as e:
                print(f"  └─ Error removing {name}: {e}")


if __name__ == "__main__":
    run_parallel_eval_suite()
