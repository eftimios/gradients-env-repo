"""
Rollout Function implementation for the Affine GAME Environment - Goofspiel Tournament.
This implementation extends the baseline with full-episode training and action masking.

Notes from tournament requirements:
- Environment: Goofspiel (Affine GAME)
- Evaluation: 250 episodes against MCTS opponent
- Opponent types available: 'random' and 'mcts' (training against 'mcts' recommended)
- No SFT or custom datasets allowed

The design includes:
- Full episode training (not just first turn)
- Action masking to optimize only on action tokens
- Context length management to prevent overflow
- BPE mismatch handling for tokenization consistency

Read more about rollout functions: https://huggingface.co/docs/trl/main/en/openenv
"""


def goofspiel_rollout_full_episode(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json

    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
    MAX_PROMPT_LEN = 24576      # Max prompt tokens before ending episode early

    # --- Game Configuration ---
    GAMES_TO_TASK_ID_RANGE = {
        "goofspiel": (0, 99999999),
        "liars_dice": (100000000, 199999999),
        "leduc_poker": (200000000, 299999999),
        "gin_rummy": (300000000, 399999999),
        "othello": (400000000, 499999999),
        "backgammon": (500000000, 599999999),
        "hex": (600000000, 699999999),
        "clobber": (700000000, 799999999),
    }
    
    SELECTED_GAME = "goofspiel"
    OPPONENT_TYPE = "mcts"  # Training against MCTS as recommended

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(goofspiel_rollout_full_episode, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        
        # Get env server for that local rank
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]
        
        if not server_list:
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]
        
        goofspiel_rollout_full_episode.base_url = base_url
        
        # Initialize environment with first reset
        try:
            print(f"Initializing Goofspiel environment on rank {rank} at {base_url}...")
            payload = {
                "task_id": GAMES_TO_TASK_ID_RANGE[SELECTED_GAME][0],
                "seed": 42,
                "opponent": OPPONENT_TYPE
            }
            reset_res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
            reset_res.raise_for_status()
            goofspiel_rollout_full_episode.initialized = True
            print(f"Goofspiel environment initialized on rank {rank}.")
        except Exception as e:
            print(f"CRITICAL: Failed to initialize environment on rank {rank}: {e}")
            raise e
    
    # Retrieve static variables
    env_endpoint = goofspiel_rollout_full_episode.base_url

    # --- 2. Rollout Setup ---
    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []
    all_episode_action_masks: list[list[int]] = []

    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    
    task_id_min, task_id_max = GAMES_TO_TASK_ID_RANGE[SELECTED_GAME]

    # --- 3. Batch Loop ---
    # Sample random game_id from the valid range for this game
    game_id = random.randint(task_id_min, task_id_max)

    for i, prompt in enumerate(prompts):
        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        episode_action_mask: list[int] = []
        prev_full_ids: list[int] | None = None
        done = False
        train_reward = 0.0
        turn_number = 0
        episode_id = ""
        
        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": OPPONENT_TYPE}
        
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data.get("result", {})
            
            # Get episode id for subsequent interactions
            episode_id = result_block.get("episode_id", "")
            
            # Construct Initial Observation with format instructions
            current_observation = result_block.get("observation", "")
            format_instructions = 'Your output must strictly follow this format: "Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."'
            formatted_observation = current_observation + "\n" + format_instructions
            
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue

        # --- Build Conversation History ---
        messages = []
        messages.append({"role": "user", "content": formatted_observation})

        # --- Interaction Loop ---
        while not done and (turn_number < max_turns):
            # Generate Rollout Completion
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Check if prompt exceeds max length - end episode early to prevent context overflow
            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}, ending episode early")
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    # BPE mismatch - tokenizer produced different IDs for same prefix text
                    # Graceful fallback: skip delta masking for this turn, just add completion
                    print(
                        f"Warning: BPE mismatch at turn {turn_number} (expected prefix {len(prev_full_ids)}, "
                        f"got {len(prompt_ids)} tokens). Skipping delta mask for this turn."
                    )
                    # Reset prev_full_ids to current prompt to try to recover alignment
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta_prompt_ids = prompt_ids[len(prev_full_ids) :]
                    if delta_prompt_ids:
                        episode_completion_ids.extend(delta_prompt_ids)
                        episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                        episode_action_mask.extend([0] * len(delta_prompt_ids))
                    prev_full_ids = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse Action ---
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            # Parse ReAct format (extract action ID)
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()
            
            # --- Step Environment (POST /step) ---
            step_reward = 0.0
            step_done = False

            try:
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data.get("result", {})

                # Extract response data
                step_state = step_block.get("observation", "")
                step_reward = step_block.get("reward", 0.0)
                step_done = step_block.get("done", False)
                
                # Format next observation
                formatted_observation = step_state
                
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = f"Invalid Action.\n\n{formatted_observation}"
                step_reward = -0.01  # Small penalty for invalid actions
                step_done = False

            # Update Loop State
            done = step_done
            if done:
                train_reward = step_reward

            if not done:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids)}), truncating")
            episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
            episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
            episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

        # For Goofspiel, reward is provided directly by environment at episode end
        all_episode_prompt_ids.append(episode_prompt_ids)
        all_episode_completion_ids.append(episode_completion_ids)
        all_episode_logprobs.append(episode_logprobs)
        all_episode_rewards.append(train_reward)
        all_episode_action_masks.append(episode_action_mask)

    return {
        "prompt_ids": all_episode_prompt_ids,
        "completion_ids": all_episode_completion_ids,
        "logprobs": all_episode_logprobs,
        "env_rewards": all_episode_rewards,
        "action_mask": all_episode_action_masks
    }


def goofspiel_rollout_reward_func(completions, **kwargs):
    """Reward function for Goofspiel that extracts rewards from rollout kwargs."""
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)

