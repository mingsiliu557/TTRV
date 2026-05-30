# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import (
    post_test_time_train_metrics, test_time_train_metrics)


class TTRLRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        reward_fn_key="data_source",
        compute_score=None,
        n_votes_per_prompt=1,
        n_samples_per_prompt=1,
        mode="eval",
        eval_n_samples=1,
        reward_style="frequency_entropy",
        soft_label_gamma=2.0,
        unknown_reward=0.0,
        all_unknown_reward=0.0,
        entropy_coef=0.75,
        answer_parse_mode="legacy",
        harmony_transform_type="photometric",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        self.reward_style = reward_style
        self.soft_label_gamma = soft_label_gamma
        self.unknown_reward = unknown_reward
        self.all_unknown_reward = all_unknown_reward
        self.entropy_coef = entropy_coef
        self.answer_parse_mode = answer_parse_mode
        self.harmony_transform_type = harmony_transform_type
        self._train_dump_call = 0
        self._eval_dump_call = 0
        assert n_votes_per_prompt >= n_samples_per_prompt, f"For TTRL settings, n_votes_per_prompt {n_votes_per_prompt} should be greater than or equal to n_samples_per_prompt {n_samples_per_prompt}"

        print(
            "TTRLRewardManager initialized with "
            f"n_votes_per_prompt {n_votes_per_prompt}, "
            f"n_samples_per_prompt {n_samples_per_prompt}, "
            f"eval_n_samples {eval_n_samples}, "
            f"reward_style {reward_style}, "
            f"soft_label_gamma {soft_label_gamma}, "
            f"answer_parse_mode {answer_parse_mode}, "
            f"harmony_transform_type {harmony_transform_type}"
        )


    def _data_source_to_task(self, data_source):
        if data_source in ["MATH-TTT", "AIME-TTT", "AMC-TTT","data/AIME-TTT"]:
            return "math"
        elif data_source in ["GPQA-TTT"]:
            return "gpqa"
        elif data_source in ["bbox"]:
            return "bbox"
        else:
            raise NotImplementedError(f"Data source {data_source} is not supported for TTRLRewardManager")

    def _write_jsonl(self, output_path, records):
        if not output_path:
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _write_eval_outputs(self, records):
        self._write_jsonl(os.environ.get("TTRL_EVAL_OUTPUT_JSONL"), records)

    def _prediction_summary(self, predictions, ground_truth):
        prediction_counter = Counter(str(prediction) for prediction in predictions)
        if prediction_counter:
            majority_prediction, majority_count = prediction_counter.most_common(1)[0]
        else:
            majority_prediction, majority_count = None, 0
        correct_count = prediction_counter.get(str(ground_truth), 0)
        return {
            "prediction_counter": dict(prediction_counter),
            "majority_prediction": majority_prediction,
            "majority_count": int(majority_count),
            "majority_ratio": float(majority_count / len(predictions)) if predictions else 0.0,
            "correct_count": int(correct_count),
            "correct_present": bool(correct_count > 0),
            "correct_is_minority": bool(correct_count > 0 and correct_count < majority_count),
        }

    def _write_train_rollouts(self, record):
        self._write_jsonl(os.environ.get("TTRL_TRAIN_ROLLOUT_JSONL"), [record])

    def _write_eval_groups(self, records):
        output_path = os.environ.get("TTRL_EVAL_GROUP_OUTPUT_JSONL")
        if not output_path or self.eval_n_samples <= 1:
            return

        grouped_records = []
        prompt_num = len(records) // self.eval_n_samples
        for prompt_i in range(prompt_num):
            group = records[prompt_i * self.eval_n_samples: (prompt_i + 1) * self.eval_n_samples]
            if not group:
                continue
            predictions = [record.get("prediction") for record in group]
            summary = self._prediction_summary(predictions, group[0].get("ground_truth"))
            grouped_records.append(
                {
                    "eval_dump_call": self._eval_dump_call,
                    "group_index": prompt_i,
                    "index": group[0].get("index"),
                    "data_source": group[0].get("data_source"),
                    "ground_truth": group[0].get("ground_truth"),
                    "prompt": group[0].get("prompt"),
                    "extra_info": group[0].get("extra_info"),
                    "single_response_raw": group[0].get("response_raw"),
                    "single_prediction": group[0].get("prediction"),
                    "single_correct": group[0].get("correct"),
                    **summary,
                    "samples": [
                        {
                            "sample_index": sample_i,
                            "response_raw": record.get("response_raw"),
                            "prediction": record.get("prediction"),
                            "reward": record.get("reward"),
                            "correct": record.get("correct"),
                            "response_token_len": record.get("response_token_len"),
                        }
                        for sample_i, record in enumerate(group)
                    ],
                }
            )

        self._write_jsonl(output_path, grouped_records)
        self._eval_dump_call += 1

    def _decode_response(self, response_idx, attention_mask, prompt_length):
        valid_response_length = attention_mask[prompt_length:].sum()
        valid_response_idx = response_idx[:valid_response_length]
        return self.tokenizer.decode(valid_response_idx, skip_special_tokens=False), int(valid_response_length)

    def compute_post_ttrl_metrics(self, data: DataProto):
        """
        Compute post TTRL metrics for the given data.
        """
        assert len(data) % self.n_samples_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_samples_per_prompt}"
        prompt_num = len(data) // self.n_samples_per_prompt

        post_ttrl_info = {}
        post_ttrl_metrics_list = defaultdict(list)

        for prompt_i in range(prompt_num):
            group_vote_rewards = []
            group_pred_outputs = []
            group_labels = []
            group_extra_info = []
            task = None

            for i in range(self.n_samples_per_prompt):
                data_item = data[prompt_i * self.n_samples_per_prompt + i]
                prompt_idx = data_item.batch["prompts"]
                prompt_length = prompt_idx.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                response_idx = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = response_idx[:valid_response_length]
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                vote_reward = data_item.batch["acc"]
                extra_info = data_item.non_tensor_batch["extra_info"]
                if task is None:
                    task = self._data_source_to_task(data_source)
                else:
                    if task != self._data_source_to_task(data_source):
                        raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_vote_rewards.append(vote_reward)
                group_extra_info.append(extra_info)

            post_ttrl_metrics = post_test_time_train_metrics(group_pred_outputs, group_labels, group_vote_rewards, task=task, extra_info=group_extra_info)
            for k, v in post_ttrl_metrics.items():
                post_ttrl_metrics_list[k].append(v)

        for k, v in post_ttrl_metrics_list.items():
            if isinstance(v, list):
                v = np.mean(v)
                print(f"[{k}]", v)
                post_ttrl_info[k] = v
        return post_ttrl_info

    def _compute_ttrl_reward(self, data: DataProto):

            reward_extra_info = defaultdict(list)
            ttrl_info = {}

            assert len(data) % self.n_votes_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_votes_per_prompt}"
            
            prompt_num = len(data) // self.n_votes_per_prompt

            reward_tensor = torch.zeros_like(data.batch["responses"][:prompt_num*self.n_samples_per_prompt], dtype=torch.float32)

            already_print_data_sources = {}

            all_ttrl_metrics = defaultdict(list)

            scores = [0.0 for _ in range(len(data))]
            
            for prompt_i in range(prompt_num):
                group_pred_outputs = []
                group_transform_outputs = []
                group_labels = []
                group_extra_info = []
                group_response_lengths = []
                group_transform_response_lengths = []
                group_transform_metadata = []
                task = None
                group_prompt_str = None
                group_data_source = None

                for i in range(self.n_votes_per_prompt):
                    data_item = data[prompt_i * self.n_votes_per_prompt + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]

                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    if self.reward_style == "vision_self_harmony":
                        if "harmony_transform_responses" not in data_item.batch:
                            raise ValueError("vision_self_harmony requires harmony_transform_responses in batch")
                        transform_response_str, transform_response_length = self._decode_response(
                            data_item.batch["harmony_transform_responses"],
                            data_item.batch["harmony_transform_attention_mask"],
                            prompt_length,
                        )
                        group_transform_outputs.append(transform_response_str)
                        group_transform_response_lengths.append(transform_response_length)
                        group_transform_metadata.append(
                            data_item.non_tensor_batch.get("harmony_transform_metadata", {})
                        )
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if group_prompt_str is None:
                        group_prompt_str = prompt_str
                    if group_data_source is None:
                        group_data_source = data_source
                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_extra_info.append(extra_info)
                    group_response_lengths.append(int(valid_response_length))
                metric_result = test_time_train_metrics(
                    group_pred_outputs,
                    group_labels,
                    task=task,
                    extra_info=group_extra_info,
                    reward_style=self.reward_style,
                    soft_label_gamma=self.soft_label_gamma,
                    unknown_reward=self.unknown_reward,
                    all_unknown_reward=self.all_unknown_reward,
                    entropy_coef=self.entropy_coef,
                    answer_parse_mode=self.answer_parse_mode,
                    transform_solutions=group_transform_outputs if self.reward_style == "vision_self_harmony" else None,
                    transform_extra_info=group_extra_info if self.reward_style == "vision_self_harmony" else None,
                    return_details=self.reward_style == "vision_self_harmony",
                )
                if self.reward_style == "vision_self_harmony":
                    rewards, ttrl_metrics, harmony_details = metric_result
                    predictions = harmony_details["original_answers"]
                    transform_predictions = harmony_details["transform_answers"]
                else:
                    rewards, ttrl_metrics = metric_result
                    harmony_details = {}
                    predictions = auto_extract(task, group_pred_outputs, extra_info=group_extra_info)
                    transform_predictions = []
                summary = self._prediction_summary(predictions, group_labels[0] if group_labels else None)
                rollout_record = {
                    "train_dump_call": self._train_dump_call,
                    "prompt_group_index": prompt_i,
                    "index": group_extra_info[0].get("index") if group_extra_info and isinstance(group_extra_info[0], dict) else None,
                    "data_source": group_data_source,
                    "ground_truth": group_labels[0] if group_labels else None,
                    "prompt": group_prompt_str,
                    "extra_info": group_extra_info[0] if group_extra_info else None,
                    "reward_style": self.reward_style,
                    "entropy_coef": self.entropy_coef,
                    "answer_parse_mode": self.answer_parse_mode,
                    **summary,
                    "samples": [
                        {
                            "sample_index": i,
                            "used_for_update": bool(i < self.n_samples_per_prompt),
                            "response_raw": group_pred_outputs[i],
                            "prediction": predictions[i],
                            "reward": float(rewards[i]),
                            "response_token_len": group_response_lengths[i],
                        }
                        for i in range(self.n_votes_per_prompt)
                    ],
                }
                if self.reward_style == "vision_self_harmony":
                    rollout_record.update(
                        {
                            "harmony_label": harmony_details["harmony_label"],
                            "harmonic_scores": harmony_details["harmonic_scores"],
                            "original_counter": harmony_details["original_counter"],
                            "transform_counter": harmony_details["transform_counter"],
                            "original_majority": harmony_details["original_majority"],
                            "transform_majority": harmony_details["transform_majority"],
                            "harmony_label_correct": bool(harmony_details["harmony_label_accuracy"]),
                            "harmony_metrics": {
                                key: harmony_details[key]
                                for key in [
                                    "paired_prediction_agreement",
                                    "distribution_tv_distance",
                                    "original_entropy",
                                    "transform_entropy",
                                    "harmony_score_max",
                                    "harmony_score_margin",
                                    "original_invalid_ratio",
                                    "transform_invalid_ratio",
                                ]
                            },
                            "harmony_transform_metadata": group_transform_metadata[0] if group_transform_metadata else {},
                            "transform_samples": [
                                {
                                    "sample_index": i,
                                    "response_raw": group_transform_outputs[i],
                                    "prediction": transform_predictions[i],
                                    "response_token_len": group_transform_response_lengths[i],
                                    "transform_metadata": group_transform_metadata[i] if i < len(group_transform_metadata) else {},
                                }
                                for i in range(self.n_votes_per_prompt)
                            ],
                        }
                    )
                self._write_train_rollouts(rollout_record)
                self._train_dump_call += 1

                for k, v in ttrl_metrics.items():
                    all_ttrl_metrics[k].append(v)

                for i in range(self.n_votes_per_prompt):
                    if i < self.n_samples_per_prompt:
                        reward_tensor[prompt_i * self.n_samples_per_prompt + i, valid_response_length - 1] = rewards[i]
                    scores[prompt_i * self.n_votes_per_prompt + i] = rewards[i]

                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", prompt_str)
                        print("[response]", response_str)
                        print("[score]", rewards[i])

            data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)
            
            for k, v in all_ttrl_metrics.items():
                if isinstance(v, list):
                    v = np.mean(v)
                    print(f"[{k}]", v)
                    ttrl_info[k] = v
            return reward_tensor, reward_extra_info, ttrl_info

    def _compute_eval_reward(self, data: DataProto):

            reward_extra_info = defaultdict(list)
            ttrl_info = {}

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            group_pred_outputs = []
            group_labels = []
            group_extra_info = []
            eval_records = []
            already_print_data_sources = {}
            task = None
            for i in range(len(data)):
                data_item = data[i]
                prompt_idx = data_item.batch["prompts"]
                prompt_length = prompt_idx.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                response_idx = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = response_idx[:valid_response_length]

                prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]
                
                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_extra_info.append(extra_info)
                eval_records.append(
                    {
                        "eval_dump_call": self._eval_dump_call,
                        "batch_index": i,
                        "index": extra_info.get("index") if isinstance(extra_info, dict) else None,
                        "data_source": data_source,
                        "ground_truth": ground_truth,
                        "prompt": prompt_str,
                        "response_raw": response_str,
                        "response_token_len": int(valid_response_length),
                        "extra_info": extra_info,
                    }
                )

                if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", prompt_str)
                        print("[response]", response_str)
                if task is None:
                    task = self._data_source_to_task(data_source)
                else:
                    if task != self._data_source_to_task(data_source):
                        raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

            rewards, verify_extra_info = auto_verify(task, group_pred_outputs, group_labels, extra_info=group_extra_info)
            predictions = auto_extract(task, group_pred_outputs, extra_info=group_extra_info)
            for record, reward, prediction in zip(eval_records, rewards, predictions):
                record["prediction"] = prediction
                record["reward"] = float(reward)
                record["correct"] = bool(reward)
            self._write_eval_outputs(eval_records)
            self._write_eval_groups(eval_records)
            if not os.environ.get("TTRL_EVAL_GROUP_OUTPUT_JSONL") or self.eval_n_samples <= 1:
                self._eval_dump_call += 1

            for k, v in verify_extra_info.items():
                if isinstance(v, list):
                    reward_extra_info[k] += v

            for i in range(len(data)):
                reward_tensor[i, valid_response_length - 1] = rewards[i]

            # Compute TTRL metrics
            all_ttrl_metrics = defaultdict(list)
            prompt_num = len(data) // self.eval_n_samples
            for prompt_i in range(prompt_num):
                group_pred_outputs_ttrl = []
                group_labels_ttrl = []
                group_extra_info_ttrl = []
                task = None

                for i in range(self.eval_n_samples):
                    data_item = data[prompt_i * self.eval_n_samples + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]

                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels_ttrl.append(ground_truth)
                    group_pred_outputs_ttrl.append(response_str)
                    group_extra_info_ttrl.append(extra_info)
                
                _, ttrl_metrics = test_time_train_metrics(group_pred_outputs_ttrl, group_labels_ttrl, task=task, extra_info=group_extra_info_ttrl)
                for k, v in ttrl_metrics.items():
                    all_ttrl_metrics[k].append(v)
            
            for k, v in all_ttrl_metrics.items():
                if isinstance(v, list):
                    v = np.mean(v)
                    print(f"[{k}]", v)
                    ttrl_info[k] = v
            
            return reward_tensor, reward_extra_info, ttrl_info

    def __call__(self, data: DataProto, return_dict=False):

        if self.mode == "train":
            # print("train reward")
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
        elif self.mode == "eval":
            # print("eval reward")
            reward_tensor, reward_extra_info, ttrl_info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported for TTRLRewardManager")

        if return_dict:
            return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                    "ttrl_info": ttrl_info,
                }
        else:
            return reward_tensor
