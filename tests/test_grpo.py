"""GRPO 概率、组回报、市场采样与训练/评估集成测试。"""

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.distributions import Normal
import yaml

from electricity_market_sim.config import load_market_settings
from electricity_market_sim.deep_learning.agent import MATD3Agent
from electricity_market_sim.deep_learning.grpo_agent import (
    GRPOAgent, action_log_prob, collect_group, group_advantages,
)
from electricity_market_sim.deep_learning.grpo_sampling import collect_parallel_group
from electricity_market_sim.deep_learning.learning_metrics import _metrics
from electricity_market_sim.deep_learning.test import evaluate_learning_scenario, run_learning_test
from electricity_market_sim.deep_learning.training import train_learning_scenario
from electricity_market_sim.errors import InputValidationError
from electricity_market_sim.main import main
from electricity_market_sim.simulation import LearningEpisodeRunner
from test_learning import make_scenario, read_rows, START, HOUR


def make_grpo_scenario(path):
    path = make_scenario(path)
    config_path = path / "config.yaml"
    scene = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    periods = scene["base"]["learning_config"]["periods"]
    scene["base"]["learning_config"] = {
        "learning_mode": True, "algorithm": "grpo", "periods": periods,
        "training_steps": 72, "group_size": 12, "update_epochs": 2, "sampling_workers": 1,
    }
    config_path.write_text(yaml.safe_dump(scene), encoding="utf-8")
    return path


class GRPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def test_transformed_density_uses_jacobian_and_joint_probability(self):
        latent = torch.tensor([0.8, -1.4], dtype=torch.float64, requires_grad=True)
        distribution = Normal(torch.tensor([0.2, 0.3]), torch.tensor([0.4, 0.6]))
        action = torch.nn.functional.softsign(latent)
        derivative, = torch.autograd.grad(action.sum(), latent)
        expected = distribution.log_prob(latent).sum() - derivative.abs().log().sum()
        torch.testing.assert_close(action_log_prob(distribution, latent), expected)

    def test_returns_stop_at_terminal_and_group_advantages_are_relative(self):
        rewards = torch.tensor([[1., 2., 100.], [3., 4., 100.]])
        dones = torch.tensor([[False, True, True], [False, True, True]])
        advantages, active = group_advantages(rewards, dones, 0.5, 1e-6)
        # 前两步回报为 [2,2] / [5,4]，第三步相同，标准差为零。
        torch.testing.assert_close(advantages, torch.tensor([[-1., -1., 0.], [1., 1., 0.]]))
        self.assertEqual(active.tolist(), [True, True, False])

    def test_group_freezes_policy_and_preserves_unsorted_actions_and_final_reward(self):
        with TemporaryDirectory() as temporary:
            path = make_grpo_scenario(Path(temporary))
            cfg = load_market_settings(path / "config.yaml").learning_config
            self.assertEqual(cfg.group_size, 12)
            agent = GRPOAgent(cfg, seed=7)
            before = {name: value.clone() for name, value in agent.actor.state_dict().items()}
            batch, results = collect_group(agent, LearningEpisodeRunner(path, period="train"))
            self.assertEqual(batch.states.shape, (12, 3, 16))
            torch.testing.assert_close(batch.states[:, 0], batch.states[0, 0].expand(12, -1))
            self.assertTrue(batch.dones[:, -1].all())
            self.assertFalse(batch.dones[:, :-1].any())
            self.assertFalse(torch.equal(batch.latent[0], batch.latent[1]))
            self.assertTrue(batch.can_bid.all())
            for i, result in enumerate(results):
                observations = torch.tensor([step.state for step in result.learning_transitions])
                torch.testing.assert_close(batch.states[i], observations)
                actions = torch.tensor([step.action for step in result.learning_steps])
                torch.testing.assert_close(torch.nn.functional.softsign(batch.latent[i]), actions)
                self.assertEqual(batch.rewards[i, -1].item(), result.learning_steps[-1].reward)
            for name, value in agent.actor.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            old = action_log_prob(Normal(batch.old_mean, batch.old_std), batch.latent)
            torch.testing.assert_close((old - batch.old_log_prob).exp(), torch.ones_like(old))
            diagnostic = agent.update(batch)
            self.assertEqual(diagnostic["interaction_steps"], 36)
            self.assertGreater(agent.gradient_updates, 0)
            self.assertTrue(any(not torch.equal(value, before[name])
                                for name, value in agent.actor.state_dict().items()))

    def test_inactive_hours_do_not_update_policy_but_count_as_interactions(self):
        with TemporaryDirectory() as temporary:
            path = make_grpo_scenario(Path(temporary))
            cfg = load_market_settings(path / "config.yaml").learning_config
            agent = GRPOAgent(cfg, seed=7)
            batch, _ = collect_group(agent, LearningEpisodeRunner(path, period="train"))
            batch.can_bid.fill_(False)
            before = {name: value.clone() for name, value in agent.actor.state_dict().items()}
            diagnostic = agent.update(batch)
            self.assertEqual(agent.interaction_steps, 36)
            self.assertEqual(agent.gradient_updates, 0)
            self.assertEqual(diagnostic["eligible_samples"], 0)
            self.assertEqual(diagnostic["sample_coverage"], 0)
            self.assertEqual(diagnostic["stop_reason"], "no_valid_samples")
            for name, value in agent.actor.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_accumulated_gradient_matches_full_batch_including_partial_minibatch(self):
        with TemporaryDirectory() as temporary:
            path = make_grpo_scenario(Path(temporary))
            cfg = load_market_settings(path / "config.yaml").learning_config
            cfg = replace(cfg, update_epochs=1)
            chunked = GRPOAgent(replace(cfg, minibatch_size=8), seed=7)
            whole = GRPOAgent(replace(cfg, minibatch_size=36), seed=7)
            batch, _ = collect_group(chunked, LearningEpisodeRunner(path, period="train"))
            # 35 个有效样本，分批时最后一批为 3 个；无效动作不参与梯度。
            batch.can_bid[0, 0] = False
            diagnostic = chunked.update(batch)
            whole.update(batch)
            self.assertEqual(diagnostic["eligible_samples"], 35)
            self.assertEqual(diagnostic["sample_coverage"], 1)
            self.assertEqual(diagnostic["updates_this_group"], 1)
            self.assertEqual(diagnostic["stop_reason"], "update_epochs")
            for name, value in chunked.actor.state_dict().items():
                torch.testing.assert_close(value, whole.actor.state_dict()[name], rtol=1e-5, atol=5e-7)

    def test_kl_stops_after_complete_group_gradient_and_reports_updated_policy_kl(self):
        with TemporaryDirectory() as temporary:
            path = make_grpo_scenario(Path(temporary))
            cfg = load_market_settings(path / "config.yaml").learning_config
            agent = GRPOAgent(replace(cfg, minibatch_size=8, target_kl=1e-10), seed=7)
            batch, _ = collect_group(agent, LearningEpisodeRunner(path, period="train"))
            diagnostic = agent.update(batch)
            self.assertEqual(diagnostic["updates_this_group"], 1)
            self.assertEqual(diagnostic["sample_coverage"], 1)
            self.assertEqual(diagnostic["stop_reason"], "target_kl")
            with torch.no_grad():
                expected = torch.distributions.kl_divergence(
                    Normal(batch.old_mean, batch.old_std), agent.actor.distribution(batch.states)
                ).sum(-1).mean().item()
            self.assertAlmostEqual(diagnostic["kl"], expected, places=6)
            self.assertGreater(diagnostic["kl"], agent.config.target_kl)

    def test_learning_rate_decay_uses_command_line_group_budget(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_grpo_scenario(root / "input")
            cfg = load_market_settings(path / "config.yaml").learning_config
            output = root / "output"
            # 场景默认两组，命令行覆盖为三组；首尾学习率按实际预算计算。
            train_learning_scenario(path, output, independent_runs=1, training_steps=108)
            diagnostics = read_rows(output / "grpo_diagnostics.csv")
            self.assertEqual(len(diagnostics), 3)
            for row, fraction in zip(diagnostics, (1.0, (1 + cfg.min_lr_ratio) / 2, cfg.min_lr_ratio)):
                self.assertAlmostEqual(float(row["learning_rate"]), cfg.learning_rate * fraction)

    def test_parallel_groups_match_serial_sampling_rng_and_policy_updates(self):
        with TemporaryDirectory() as temporary:
            path = make_grpo_scenario(Path(temporary))
            cfg = load_market_settings(path / "config.yaml").learning_config
            serial = GRPOAgent(cfg, seed=7, run=1)
            parallel = GRPOAgent(replace(cfg, sampling_workers=2), seed=7, run=1)
            runner = LearningEpisodeRunner(path, period="train")
            with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
                for group in range(2):
                    serial_metrics = []
                    batch, _ = collect_group(serial, runner, on_episode=lambda result: serial_metrics.append(
                        _metrics(result, run=1, phase="train", gamma=cfg.gamma,
                                 episode=group * cfg.group_size + len(serial_metrics) + 1)
                    ))
                    other, metrics = collect_parallel_group(
                        parallel, runner, pool, steps_per_episode=3, first_episode=group * cfg.group_size
                    )
                    for name in batch.__dataclass_fields__:
                        torch.testing.assert_close(getattr(batch, name), getattr(other, name), rtol=0, atol=0)
                    self.assertEqual(serial_metrics, metrics)
                    torch.testing.assert_close(serial.rng.get_state(), parallel.rng.get_state(), rtol=0, atol=0)
                    self.assertEqual(serial.update(batch), parallel.update(other))
                    for name, value in serial.actor.state_dict().items():
                        torch.testing.assert_close(value, parallel.actor.state_dict()[name], rtol=0, atol=0)

    def test_parallel_training_and_testing_match_serial_results(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_grpo_scenario(root / "input")
            serial, parallel = root / "serial", root / "parallel"
            train_learning_scenario(path, serial, independent_runs=1)
            config_path = path / "config.yaml"
            scene = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            scene["base"]["learning_config"]["sampling_workers"] = 2
            config_path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            train_learning_scenario(path, parallel, independent_runs=1)
            self.assertEqual((serial / "learning_metrics.csv").read_bytes(),
                             (parallel / "learning_metrics.csv").read_bytes())
            original = torch.load(serial / "checkpoints" / "best.pt", weights_only=True)
            candidate = torch.load(parallel / "checkpoints" / "best.pt", weights_only=True)
            self.assertEqual(original["episode"], candidate["episode"])
            for name, value in original["actor"].items():
                torch.testing.assert_close(value, candidate["actor"][name], rtol=0, atol=0)
            run_learning_test(path, serial)
            run_learning_test(path, parallel)
            self.assertEqual((serial / "learning_test_metrics.csv").read_bytes(),
                             (parallel / "learning_test_metrics.csv").read_bytes())

    def test_train_saves_only_policy_and_test_is_read_only(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_grpo_scenario(root / "input")
            output = root / "output"
            train_learning_scenario(path, output, independent_runs=1)
            best = output / "checkpoints" / "best.pt"
            payload = torch.load(best, weights_only=True)
            self.assertIn("actor", payload)
            for field in ("optimizer", "rng_state", "training_config"):
                self.assertNotIn(field, payload)
            self.assertFalse((output / "checkpoints" / "latest.pt").exists())
            self.assertFalse((output / "test").exists())
            before = {file.name: file.read_bytes() for file in (output / "checkpoints").iterdir()}
            result = run_learning_test(path, output)
            self.assertEqual([step.delivery_start for step in result.learning_steps],
                             [START + 6 * HOUR, START + 7 * HOUR])
            self.assertEqual(before, {file.name: file.read_bytes()
                                     for file in (output / "checkpoints").iterdir()})
            self.assertEqual(len(read_rows(output / "grpo_diagnostics.csv")), 2)
            self.assertEqual([int(row["interaction_steps"])
                              for row in read_rows(output / "learning_progress.csv")], [36, 72])
            cfg = load_market_settings(path / "config.yaml").learning_config
            with self.assertRaises(InputValidationError):
                MATD3Agent(cfg, total_regular_steps=1).load_checkpoint(best)

    def test_multirun_manifest_and_deterministic_evaluation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_grpo_scenario(root / "input")
            output = root / "output"
            train_learning_scenario(path, output, independent_runs=3, training_steps=36)
            manifest_path = output / "checkpoints" / "runs.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["algorithm"], "grpo")
            self.assertEqual([entry["run"] for entry in manifest["runs"]], [1, 2, 3])
            self.assertFalse(list((output / "checkpoints").rglob("latest.pt")))
            best = output / "checkpoints" / "best.pt"
            first = evaluate_learning_scenario(path, root / "eval1", checkpoint_path=best)
            second = evaluate_learning_scenario(path, root / "eval2", checkpoint_path=best)
            self.assertEqual(first.learning_steps, second.learning_steps)
            run_learning_test(path, output)
            self.assertEqual(len(read_rows(output / "learning_test_metrics.csv")), 6)

    def test_fresh_training_ignores_incomplete_old_run_records(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_grpo_scenario(root / "input")
            output = root / "output"
            manifest_path = output / "checkpoints" / "runs.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps({
                "algorithm": "grpo", "periods": {}, "runs": [{"run": "incomplete"}],
            }), encoding="utf-8")
            train_learning_scenario(path, output, independent_runs=1, training_steps=36)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([entry["run"] for entry in manifest["runs"]], [1])
            run_learning_test(path, output)

    def test_advantage_version_is_metadata_while_action_distribution_must_match(self):
        with TemporaryDirectory() as temporary:
            path = make_grpo_scenario(Path(temporary) / "input")
            cfg = load_market_settings(path / "config.yaml").learning_config
            agent = GRPOAgent(cfg, seed=7)
            state = (0.1,) * 16
            expected = agent.deterministic_action(state)
            checkpoint = Path(temporary) / "policy.pt"
            agent.save_checkpoint(checkpoint, episode=0, validation_total_reward=0)
            payload = torch.load(checkpoint, weights_only=True)
            payload["advantage_version"] = "trajectory-return-v2"
            torch.save(payload, checkpoint)
            restored = GRPOAgent(cfg, seed=8)
            metadata = restored.load_checkpoint(checkpoint)
            self.assertEqual(metadata["advantage_version"], "trajectory-return-v2")
            self.assertEqual(restored.deterministic_action(state), expected)
            payload["distribution"] = "different-action-transform"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(InputValidationError, "Incompatible GRPO checkpoint"):
                restored.load_checkpoint(checkpoint)

    def test_training_rejects_resume_options(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_grpo_scenario(root / "input")
            with self.assertRaisesRegex(InputValidationError, "从头训练"):
                train_learning_scenario(path, root / "output", checkpoint_path=root / "old.pt")
            config_path = path / "config.yaml"
            scene = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            scene["base"]["learning_config"]["continue_learning"] = True
            config_path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            with self.assertRaisesRegex(InputValidationError, "从头训练"):
                train_learning_scenario(path, root / "output")

    def test_cli_steps_and_incomplete_group_budget(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_grpo_scenario(root / "input")
            with self.assertRaisesRegex(InputValidationError, "完整组"):
                train_learning_scenario(path, root / "bad", independent_runs=1, training_steps=35)
            with patch("builtins.print"):
                self.assertEqual(main([
                    "--input-dir", str(path), "--output-dir", str(root / "out"),
                    "--training-steps", "36", "--training-runs", "1", "--no-plots",
                ]), 0)
                self.assertEqual(main([
                    "--input-dir", str(path), "--output-dir", str(root / "out"),
                    "--learning-mode", "test", "--no-plots",
                ]), 0)

    def test_cli_rounds_use_scenario_group_size_and_delivery_hours(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_grpo_scenario(root / "input")
            config_path = path / "config.yaml"
            scene = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            scene["base"]["learning_config"]["group_size"] = 6
            config_path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            output = root / "output"
            with patch("builtins.print"):
                self.assertEqual(main([
                    "--input-dir", str(path), "--output-dir", str(output),
                    "--training-rounds", "3", "--training-runs", "1", "--no-plots",
                ]), 0)
            # 每条轨迹 3 小时、每组 6 条；3 轮共 54 次交互，覆盖原场景的 72 步预算。
            diagnostics = read_rows(output / "grpo_diagnostics.csv")
            self.assertEqual([int(row["interaction_steps"]) for row in diagnostics], [18, 36, 54])
            payload = torch.load(output / "checkpoints" / "best.pt", weights_only=True)
            self.assertIn(payload["episode"], (6, 12, 18))

    def test_matd3_step_budget_counts_warmup_interactions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            train_learning_scenario(path, root / "output", training_steps=36, independent_runs=1)
            payload = torch.load(root / "output" / "checkpoints" / "latest.pt", weights_only=True)
            self.assertEqual(payload["episode"], 12)
            self.assertEqual(len(payload["replay"]["states"]), 36)
            progress = read_rows(root / "output" / "learning_progress.csv")
            self.assertEqual(int(progress[-1]["interaction_steps"]), 36)

    def test_config_rejects_invalid_grpo_parameters_and_cross_algorithm_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_grpo_scenario(root / "input")
            config_path = path / "config.yaml"
            scene = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            for name, value in (("group_size", 1), ("sampling_workers", 0),
                                ("initial_std", 2), ("advantage_threshold", 0),
                                ("max_grad_norm", 0), ("weight_decay", -0.01), ("min_lr_ratio", 1.1)):
                invalid = yaml.safe_load(yaml.safe_dump(scene))
                invalid["base"]["learning_config"][name] = value
                config_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
                with self.assertRaises(InputValidationError):
                    load_market_settings(config_path)
            config_path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            train_learning_scenario(path, root / "output", training_steps=36, independent_runs=1)
            scene["base"]["learning_config"]["algorithm"] = "matd3"
            config_path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            with self.assertRaisesRegex(InputValidationError, "algorithm"):
                train_learning_scenario(path, root / "output", independent_runs=1)


if __name__ == "__main__":
    unittest.main()
