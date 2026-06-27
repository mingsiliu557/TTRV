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
    DENSITY_REWARD_STYLES, NUMERIC_REWARD_STYLES, _extract_answers,
    _has_choice_ground_truth, _normalize_choice_answer,
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
        feature_center_hsr_alpha=0.5,
        feature_center_hsr_beta=0.2,
        entropy_temperature_version="v3",
        entropy_temperature_tau0=0.25,
        entropy_temperature_gamma=1.0,
        entropy_temperature_lambda=0.5,
        entropy_temperature_tau_min=0.05,
        density_temperature_t0=0.2,
        density_temperature_t_min=0.05,
        density_temperature_t_max=0.8,
        density_embedding_scope="response_mean_pool",
        density_evidence_template="",
        numeric_kernel_sigma=0.15,
        numeric_trim_ratio=0.2,
        answer_choice_labels="A-D",
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
        self.feature_center_hsr_alpha = feature_center_hsr_alpha
        self.feature_center_hsr_beta = feature_center_hsr_beta
        self.entropy_temperature_version = entropy_temperature_version
        self.entropy_temperature_tau0 = entropy_temperature_tau0
        self.entropy_temperature_gamma = entropy_temperature_gamma
        self.entropy_temperature_lambda = entropy_temperature_lambda
        self.entropy_temperature_tau_min = entropy_temperature_tau_min
        self.density_temperature_t0 = density_temperature_t0
        self.density_temperature_t_min = density_temperature_t_min
        self.density_temperature_t_max = density_temperature_t_max
        self.density_embedding_scope = density_embedding_scope
        self.density_evidence_template = density_evidence_template
        self.numeric_kernel_sigma = numeric_kernel_sigma
        self.numeric_trim_ratio = numeric_trim_ratio
        self.answer_choice_labels = answer_choice_labels
        self.log_response_embeddings = os.environ.get("TTRL_LOG_RESPONSE_EMBEDDINGS", "0") == "1"
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
            f"harmony_transform_type {harmony_transform_type}, "
            f"feature_center_hsr_alpha {feature_center_hsr_alpha}, "
            f"feature_center_hsr_beta {feature_center_hsr_beta}, "
            f"entropy_temperature_version {entropy_temperature_version}, "
            f"entropy_temperature_tau0 {entropy_temperature_tau0}, "
            f"entropy_temperature_gamma {entropy_temperature_gamma}, "
            f"entropy_temperature_lambda {entropy_temperature_lambda}, "
            f"entropy_temperature_tau_min {entropy_temperature_tau_min}, "
            f"density_temperature_t0 {density_temperature_t0}, "
            f"density_temperature_t_min {density_temperature_t_min}, "
            f"density_temperature_t_max {density_temperature_t_max}, "
            f"density_embedding_scope {density_embedding_scope}, "
            f"density_evidence_template_set {bool(density_evidence_template)}, "
            f"numeric_kernel_sigma {numeric_kernel_sigma}, "
            f"numeric_trim_ratio {numeric_trim_ratio}, "
            f"answer_choice_labels {answer_choice_labels}"
        )


    def _data_source_to_task(self, data_source):
        if data_source in ["MATH-TTT", "AIME-TTT", "AMC-TTT","data/AIME-TTT"]:
            return "math"
        elif data_source in ["GPQA-TTT"]:
            return "gpqa"
        elif data_source in ["bbox"]:
            return "bbox"
        elif data_source in ["VQA-DA-TTT", "VisualSimpleQA-TTT", "AOKVQA-DA-TTT"]:
            return "vqa_da"
        elif data_source in ["OCR-TTT", "OCRBench-TTT"]:
            return "ocr"
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

            post_ttrl_metrics = post_test_time_train_metrics(
                group_pred_outputs,
                group_labels,
                group_vote_rewards,
                task=task,
                extra_info=group_extra_info,
                answer_parse_mode=self.answer_parse_mode,
                answer_choice_labels=self.answer_choice_labels,
            )
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
                group_response_embeddings = []
                group_density_evidence_queries = []
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
                    if self.reward_style in {"feature_center_hard", "feature_center_hsr"} or self.reward_style in DENSITY_REWARD_STYLES:
                        if "response_embeddings" not in data_item.batch:
                            raise ValueError(f"{self.reward_style} requires response_embeddings in batch")
                        group_response_embeddings.append(
                            data_item.batch["response_embeddings"].detach().cpu().float().numpy()
                        )
                        group_density_evidence_queries.append(
                            data_item.non_tensor_batch.get("density_evidence_query", "")
                        )
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
                    response_embeddings=(
                        group_response_embeddings
                        if self.reward_style in {"feature_center_hard", "feature_center_hsr"} or self.reward_style in DENSITY_REWARD_STYLES
                        else None
                    ),
                    feature_center_hsr_alpha=self.feature_center_hsr_alpha,
                    feature_center_hsr_beta=self.feature_center_hsr_beta,
                    entropy_temperature_version=self.entropy_temperature_version,
                    entropy_temperature_tau0=self.entropy_temperature_tau0,
                    entropy_temperature_gamma=self.entropy_temperature_gamma,
                    entropy_temperature_lambda=self.entropy_temperature_lambda,
                    entropy_temperature_tau_min=self.entropy_temperature_tau_min,
                    density_temperature_t0=self.density_temperature_t0,
                    density_temperature_t_min=self.density_temperature_t_min,
                    density_temperature_t_max=self.density_temperature_t_max,
                    density_embedding_scope=self.density_embedding_scope,
                    numeric_kernel_sigma=self.numeric_kernel_sigma,
                    numeric_trim_ratio=self.numeric_trim_ratio,
                    answer_choice_labels=self.answer_choice_labels,
                    return_details=(
                        self.reward_style in {
                            "vision_self_harmony",
                            "choice_majority_vote",
                            "frequency_valid_only",
                            "frequency_valid_entropy",
                            "entropy_temperature_frequency",
                            "numeric_kernel_density",
                            "numeric_trimmed_mean",
                            "feature_center_hard",
                            "feature_center_hsr",
                        }
                        or self.reward_style in DENSITY_REWARD_STYLES
                        or self.reward_style in NUMERIC_REWARD_STYLES
                    ),
                )
                choice_details = {}
                if self.reward_style == "vision_self_harmony":
                    rewards, ttrl_metrics, harmony_details = metric_result
                    predictions = harmony_details["original_answers"]
                    transform_predictions = harmony_details["transform_answers"]
                elif self.reward_style == "choice_majority_vote":
                    rewards, ttrl_metrics, choice_details = metric_result
                    harmony_details = {}
                    feature_center_details = {}
                    density_details = {}
                    predictions = choice_details["original_answers"]
                    transform_predictions = []
                elif self.reward_style in {"frequency_valid_only", "frequency_valid_entropy", "entropy_temperature_frequency"} or self.reward_style in NUMERIC_REWARD_STYLES:
                    rewards, ttrl_metrics, choice_details = metric_result
                    harmony_details = {}
                    feature_center_details = {}
                    density_details = {}
                    predictions = choice_details["original_answers"]
                    transform_predictions = []
                elif self.reward_style in {"feature_center_hard", "feature_center_hsr"}:
                    rewards, ttrl_metrics, feature_center_details = metric_result
                    harmony_details = {}
                    choice_details = {}
                    density_details = {}
                    predictions = feature_center_details["original_answers"]
                    transform_predictions = []
                elif self.reward_style in DENSITY_REWARD_STYLES:
                    rewards, ttrl_metrics, density_details = metric_result
                    harmony_details = {}
                    choice_details = {}
                    feature_center_details = {}
                    predictions = density_details["original_answers"]
                    transform_predictions = []
                else:
                    rewards, ttrl_metrics = metric_result
                    harmony_details = {}
                    feature_center_details = {}
                    density_details = {}
                    predictions = auto_extract(task, group_pred_outputs, extra_info=group_extra_info)
                    transform_predictions = []
                prompt_update_indices = list(range(self.n_samples_per_prompt))
                prompt_update_index_set = set(prompt_update_indices)

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
                    "feature_center_hsr_alpha": self.feature_center_hsr_alpha,
                    "feature_center_hsr_beta": self.feature_center_hsr_beta,
                    "entropy_temperature_version": self.entropy_temperature_version,
                    "entropy_temperature_tau0": self.entropy_temperature_tau0,
                    "entropy_temperature_gamma": self.entropy_temperature_gamma,
                    "entropy_temperature_lambda": self.entropy_temperature_lambda,
                    "entropy_temperature_tau_min": self.entropy_temperature_tau_min,
                    "density_temperature_t0": self.density_temperature_t0,
                    "density_temperature_t_min": self.density_temperature_t_min,
                    "density_temperature_t_max": self.density_temperature_t_max,
                    "density_embedding_scope": self.density_embedding_scope,
                    "numeric_kernel_sigma": self.numeric_kernel_sigma,
                    "numeric_trim_ratio": self.numeric_trim_ratio,
                    "answer_choice_labels": self.answer_choice_labels,
                    **summary,
                    "samples": [
                        {
                            "sample_index": i,
                            "used_for_update": bool(i in prompt_update_index_set),
                            "response_raw": group_pred_outputs[i],
                            "prediction": predictions[i],
                            "reward": float(rewards[i]),
                            "response_token_len": group_response_lengths[i],
                            **({
                                "p_reward": choice_details.get("p_rewards", [None] * self.n_votes_per_prompt)[i],
                                "q_reward": choice_details.get("q_rewards", [None] * self.n_votes_per_prompt)[i],
                            } if self.reward_style == "entropy_temperature_frequency" else {}),
                            **({
                                "numeric_value": choice_details.get("numeric_values", [None] * self.n_votes_per_prompt)[i],
                                "numeric_reward": choice_details.get("numeric_rewards", [None] * self.n_votes_per_prompt)[i],
                            } if self.reward_style in NUMERIC_REWARD_STYLES else {}),
                            **({
                                "evidence": feature_center_details.get("evidences", [""] * self.n_votes_per_prompt)[i],
                                "feature_center_distance": feature_center_details.get("feature_center_distances", [None] * self.n_votes_per_prompt)[i],
                                "hsr_hard_reward": feature_center_details.get("hsr_hard_rewards", [None] * self.n_votes_per_prompt)[i],
                                "hsr_jaccard_reward": feature_center_details.get("hsr_jaccard_rewards", [None] * self.n_votes_per_prompt)[i],
                                "hsr_embedding_reward": feature_center_details.get("hsr_embedding_rewards", [None] * self.n_votes_per_prompt)[i],
                            } if self.reward_style in {"feature_center_hard", "feature_center_hsr"} else {}),
                            **({
                                "evidence": density_details.get("evidences", [""] * self.n_votes_per_prompt)[i],
                                "evidence_query": group_density_evidence_queries[i] if i < len(group_density_evidence_queries) else "",
                                "density_embedding_scope": density_details.get("density_embedding_scope", self.density_embedding_scope),
                                "density_reward": density_details.get("density_rewards", [None] * self.n_votes_per_prompt)[i],
                                "density_hard_reward": density_details.get("density_hard_rewards", [None] * self.n_votes_per_prompt)[i],
                                "density_soft_reward": density_details.get("density_soft_rewards", [None] * self.n_votes_per_prompt)[i],
                                "density_cluster_reward": density_details.get("density_cluster_rewards", [None] * self.n_votes_per_prompt)[i],
                                "density_mass": density_details.get("density_masses", [None] * self.n_votes_per_prompt)[i],
                                "similarity_to_peak": density_details.get("similarity_to_peak", [None] * self.n_votes_per_prompt)[i],
                            } if self.reward_style in DENSITY_REWARD_STYLES else {}),
                            **({
                                "response_embedding": group_response_embeddings[i].tolist(),
                            } if self.log_response_embeddings and i < len(group_response_embeddings) else {}),
                        }
                        for i in range(self.n_votes_per_prompt)
                    ],
                }
                if self.reward_style == "choice_majority_vote":
                    rollout_record.update(
                        {
                            "choice_majority_label": choice_details["choice_majority_label"],
                            "choice_counter": choice_details["choice_counter"],
                            "choice_valid_ratio": choice_details["choice_valid_ratio"],
                            "choice_majority_label_correct": bool(
                                choice_details["choice_majority_label"] == (group_labels[0] if group_labels else None)
                            ),
                        }
                    )
                if self.reward_style in {"frequency_valid_only", "frequency_valid_entropy"}:
                    rollout_record.update(
                        {
                            "valid_counter": choice_details["valid_counter"],
                            "valid_vote_ratio": choice_details["valid_vote_ratio"],
                            "valid_normalized_entropy": choice_details["valid_normalized_entropy"],
                            "valid_answer_mode": choice_details.get("valid_answer_mode"),
                        }
                    )
                    if self.reward_style == "frequency_valid_entropy":
                        rollout_record["valid_entropy_coef"] = choice_details.get("entropy_coef", self.entropy_coef)
                if self.reward_style in NUMERIC_REWARD_STYLES:
                    rollout_record.update(
                        {
                            "valid_counter": choice_details["valid_counter"],
                            "valid_vote_ratio": choice_details["valid_vote_ratio"],
                            "valid_normalized_entropy": choice_details["valid_normalized_entropy"],
                            "valid_answer_mode": choice_details.get("valid_answer_mode"),
                            "numeric_reward_mode": choice_details["numeric_reward_mode"],
                            "numeric_pseudo_label": choice_details["numeric_pseudo_label"],
                            "numeric_pseudo_value": choice_details["numeric_pseudo_value"],
                            "numeric_pseudo_support": choice_details["numeric_pseudo_support"],
                            "numeric_density_max": choice_details["numeric_density_max"],
                            "numeric_density_margin": choice_details["numeric_density_margin"],
                            "numeric_unique_count": choice_details["numeric_unique_count"],
                            "numeric_top_ratio": choice_details["numeric_top_ratio"],
                        }
                    )
                if self.reward_style == "entropy_temperature_frequency":
                    rollout_record.update(
                        {
                            "valid_counter": choice_details["valid_counter"],
                            "valid_vote_ratio": choice_details["valid_vote_ratio"],
                            "valid_distribution_p": choice_details["valid_distribution_p"],
                            "softmax_distribution_q": choice_details["softmax_distribution_q"],
                            "valid_normalized_entropy": choice_details["valid_normalized_entropy"],
                            "temperature": choice_details["temperature"],
                            "temperature_version": choice_details["temperature_version"],
                            "temperature_majority_label": choice_details["majority_label"],
                            "temperature_second_label": choice_details["second_label"],
                            "temperature_reward_margin_top2": choice_details["reward_margin_top2"],
                        }
                    )
                if self.reward_style in DENSITY_REWARD_STYLES:
                    rollout_record.update(
                        {
                            "density_peak_label": density_details["density_peak_label"],
                            "density_peak_label_correct": bool(density_details["density_peak_label_accuracy"]),
                            "density_peak_sample_index": density_details["density_peak_sample_index"],
                            "density_peak_mass": density_details["density_peak_mass"],
                            "density_temperature": density_details["density_temperature"],
                            "density_temperature_mode": density_details["density_temperature_mode"],
                            "density_reward_mode": density_details["density_reward_mode"],
                            "density_embedding_scope": density_details.get("density_embedding_scope", self.density_embedding_scope),
                            "density_cluster_masses": density_details.get("density_cluster_masses", {}),
                            "density_valid_counter": density_details["density_valid_counter"],
                            "density_valid_ratio": density_details["density_valid_ratio"],
                            "density_answer_entropy": density_details["density_answer_entropy"],
                            "density_density_entropy": density_details["density_density_entropy"],
                            "density_sim_mean": density_details["density_sim_mean"],
                            "density_sim_std": density_details["density_sim_std"],
                            "density_majority": density_details["original_majority"],
                            "density_majority_correct": bool(density_details["original_majority_accuracy"]),
                            "arithmetic_centroid_label": density_details["arithmetic_centroid_label"],
                            "arithmetic_centroid_label_correct": bool(density_details["arithmetic_centroid_label_accuracy"]),
                            "arithmetic_centroid_sample_index": density_details["arithmetic_centroid_sample_index"],
                            "density_vs_majority_agreement": density_details["density_vs_majority_agreement"],
                            "density_vs_centroid_agreement": density_details["density_vs_centroid_agreement"],
                            "corr_adv_density_freq": density_details["corr_adv_density_freq"],
                            "mean_abs_diff_adv_density_freq": density_details["mean_abs_diff_adv_density_freq"],
                        }
                    )
                if self.reward_style in {"feature_center_hard", "feature_center_hsr"}:
                    rollout_record.update(
                        {
                            "feature_center_label": feature_center_details["feature_center_label"],
                            "feature_center_candidate_index": feature_center_details["feature_center_candidate_index"],
                            "feature_center_raw_candidate_index": feature_center_details["feature_center_raw_candidate_index"],
                            "feature_center_label_correct": bool(feature_center_details["feature_center_label_accuracy"]),
                            "feature_center_min_distance": feature_center_details["feature_center_min_distance"],
                            "feature_center_margin": feature_center_details["feature_center_margin"],
                            "feature_center_valid_ratio": feature_center_details["feature_center_valid_ratio"],
                            "feature_center_counter": feature_center_details["original_counter"],
                            "feature_center_majority": feature_center_details["original_majority"],
                            "pseudo_vs_majority_agreement": feature_center_details["pseudo_vs_majority_agreement"],
                            "pseudo_response_raw": feature_center_details.get("pseudo_response_raw"),
                            "pseudo_evidence": feature_center_details.get("pseudo_evidence"),
                            "hsr_component_means": {
                                "hard": feature_center_details.get("hsr_hard_mean"),
                                "jaccard": feature_center_details.get("hsr_jaccard_mean"),
                                "embedding": feature_center_details.get("hsr_embedding_mean"),
                            },
                        }
                    )
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

            predictions = _extract_answers(
                task,
                group_pred_outputs,
                group_labels[0] if group_labels else None,
                extra_info=group_extra_info,
                answer_parse_mode=self.answer_parse_mode,
                choice_labels=self.answer_choice_labels,
            )
            choice_eval = task not in {"vqa_da", "ocr"} and bool(group_labels) and _has_choice_ground_truth(
                group_labels[0],
                choice_labels=self.answer_choice_labels,
            )
            if choice_eval:
                rewards = [
                    float(
                        _normalize_choice_answer(prediction, choice_labels=self.answer_choice_labels)
                        == _normalize_choice_answer(label, choice_labels=self.answer_choice_labels)
                    )
                    for prediction, label in zip(predictions, group_labels)
                ]
                verify_extra_info = defaultdict(list)
                verify_extra_info["acc"] = rewards
                verify_extra_info["pred"] = predictions
            elif self.answer_parse_mode == "legacy":
                rewards, verify_extra_info = auto_verify(task, group_pred_outputs, group_labels, extra_info=group_extra_info)
                predictions = auto_extract(task, group_pred_outputs, extra_info=group_extra_info)
            else:
                rewards, verify_extra_info = auto_verify(task, predictions, group_labels, extra_info=group_extra_info)
            metrics = verify_extra_info.get("metric", []) if isinstance(verify_extra_info, dict) else []
            exact_acc = verify_extra_info.get("exact_acc", []) if isinstance(verify_extra_info, dict) else []
            for idx, (record, reward, prediction) in enumerate(zip(eval_records, rewards, predictions)):
                metric_name = metrics[idx] if idx < len(metrics) else ""
                record["prediction"] = prediction
                record["reward"] = float(reward)
                if metric_name:
                    record["metric"] = metric_name
                if idx < len(exact_acc):
                    record["exact_acc"] = float(exact_acc[idx])
                if metric_name == "smape":
                    exact = float(exact_acc[idx]) if idx < len(exact_acc) else 0.0
                    record["smape_score"] = float(reward)
                    record["correct"] = bool(exact)
                else:
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
                
                _, ttrl_metrics = test_time_train_metrics(
                    group_pred_outputs_ttrl,
                    group_labels_ttrl,
                    task=task,
                    extra_info=group_extra_info_ttrl,
                    answer_parse_mode=self.answer_parse_mode,
                    answer_choice_labels=self.answer_choice_labels,
                )
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
