import base64
import json
import re
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from minisweagent import package_dir
from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.run.benchmarks.swebench import (
    PrePatchApplyError,
    PrePatchMissingError,
    apply_pre_patch,
    filter_instances,
    get_swebench_docker_image_name,
    main,
    remove_from_preds_file,
    resolve_pre_patch_path,
    update_preds_file,
)
from minisweagent.run.benchmarks.utils.localization import Localization


def _make_model_from_fixture(text_outputs: list[str], cost_per_call: float = 1.0, **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from trajectory fixture data (raw text outputs)."""

    def parse_command(text: str) -> list[dict]:
        match = re.search(r"```mswea_bash_command\s*\n(.*?)\n```", text, re.DOTALL)
        return [{"command": match.group(1)}] if match else []

    return DeterministicModel(
        outputs=[make_output(text, parse_command(text), cost=cost_per_call) for text in text_outputs],
        cost_per_call=cost_per_call,
        **kwargs,
    )


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_swebench_end_to_end(github_test_data, tmp_path, workers, container_executable):
    """Test the complete SWEBench flow using the _test subset with deterministic model"""

    model_responses = github_test_data["model_responses"]

    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        # Use side_effect to create a new model instance for each worker
        mock_get_model.side_effect = lambda **kwargs: _make_model_from_fixture(model_responses, cost_per_call=0.1)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            workers=workers,
            filter_spec="swe-agent__test-repo-1",
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
            environment_class="docker",
        )

    traj_file_path = package_dir.parent.parent / "tests" / "test_data" / "github_issue.traj.json"
    trajectory = json.loads(traj_file_path.read_text())

    last_message = trajectory[-1]["content"]
    # The final submission contains localization JSON + delimiter + patch.
    _, _, expected_patch = last_message.partition("<<<MSWEA_PATCH_DELIMITER>>>")
    expected_patch = expected_patch.lstrip("\n")

    instance_id = "swe-agent__test-repo-1"

    with open(tmp_path / "preds.json") as f:
        actual_result = json.load(f)

    entry = actual_result[instance_id]
    assert entry["model_name_or_path"] == "deterministic"
    assert entry["instance_id"] == instance_id
    assert entry["model_patch"] == expected_patch
    assert entry["localization_parse_error"] is None
    assert entry["localization"] == {
        "root_cause": [
            {
                "file": "tests/missing_colon.py",
                "line": 4,
                "statement": "def division(a: float, b: float) -> float",
            }
        ],
        "reasoning": "The function definition on line 4 is missing a trailing colon, causing a SyntaxError.",
        "confidence": "high",
    }

    traj_output_file = tmp_path / instance_id / f"{instance_id}.traj.json"
    output_trajectory = json.loads(traj_output_file.read_text())
    assert output_trajectory["messages"][-1]["content"] == last_message
    assert output_trajectory["info"]["localization"] == entry["localization"]
    assert output_trajectory["info"]["submission"] == expected_patch


def test_get_image_name_with_existing_image_name():
    """Test get_image_name when image_name is already provided"""
    instance = {"image_name": "custom/image:tag", "instance_id": "test__repo__1"}
    assert get_swebench_docker_image_name(instance) == "custom/image:tag"


def test_get_image_name_without_image_name():
    """Test get_image_name when image_name needs to be constructed"""
    instance = {"instance_id": "swe-agent__test-repo__1"}
    expected = "docker.io/swebench/sweb.eval.x86_64.swe-agent_1776_test-repo_1776_1:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_get_image_name_with_none_image_name():
    """Test get_image_name when image_name is explicitly None"""
    instance = {"image_name": None, "instance_id": "django__django__4.0"}
    expected = "docker.io/swebench/sweb.eval.x86_64.django_1776_django_1776_4.0:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_get_image_name_with_complex_instance_id():
    """Test get_image_name with complex instance_id containing multiple double underscores"""
    instance = {"instance_id": "project__sub__module__version__1.2.3"}
    expected = "docker.io/swebench/sweb.eval.x86_64.project_1776_sub_1776_module_1776_version_1776_1.2.3:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_filter_instances_no_filters():
    """Test filter_instances with no filtering applied"""
    instances = [{"instance_id": "repo1__test1"}, {"instance_id": "repo2__test2"}, {"instance_id": "repo3__test3"}]
    result = filter_instances(instances, filter_spec="", slice_spec="")
    assert result == instances


def test_filter_instances_regex_filter():
    """Test filter_instances with regex filtering"""
    instances = [
        {"instance_id": "django__test1"},
        {"instance_id": "flask__test2"},
        {"instance_id": "django__test3"},
        {"instance_id": "requests__test4"},
    ]
    result = filter_instances(instances, filter_spec=r"django__.*", slice_spec="")
    expected = [{"instance_id": "django__test1"}, {"instance_id": "django__test3"}]
    assert result == expected


def test_filter_instances_slice_only():
    """Test filter_instances with slice specification"""
    instances = [{"instance_id": f"repo{i}__test{i}"} for i in range(10)]
    result = filter_instances(instances, filter_spec="", slice_spec="2:5")
    expected = [{"instance_id": "repo2__test2"}, {"instance_id": "repo3__test3"}, {"instance_id": "repo4__test4"}]
    assert result == expected


def test_filter_instances_slice_start_only():
    """Test filter_instances with slice start only"""
    instances = [{"instance_id": f"repo{i}__test{i}"} for i in range(5)]
    result = filter_instances(instances, filter_spec="", slice_spec="3:")
    expected = [{"instance_id": "repo3__test3"}, {"instance_id": "repo4__test4"}]
    assert result == expected


def test_filter_instances_slice_end_only():
    """Test filter_instances with slice end only"""
    instances = [{"instance_id": f"repo{i}__test{i}"} for i in range(5)]
    result = filter_instances(instances, filter_spec="", slice_spec=":2")
    expected = [{"instance_id": "repo0__test0"}, {"instance_id": "repo1__test1"}]
    assert result == expected


def test_filter_instances_filter_and_slice():
    """Test filter_instances with both filtering and slicing"""
    instances = [
        {"instance_id": "django__test1"},
        {"instance_id": "flask__test2"},
        {"instance_id": "django__test3"},
        {"instance_id": "django__test4"},
        {"instance_id": "requests__test5"},
    ]
    result = filter_instances(instances, filter_spec=r"django__.*", slice_spec="1:3")
    expected = [{"instance_id": "django__test3"}, {"instance_id": "django__test4"}]
    assert result == expected


def test_filter_instances_shuffle():
    """Test filter_instances with shuffle enabled produces deterministic results"""
    instances = [{"instance_id": f"repo{i:02d}__test{i}"} for i in range(10)]
    # Test that shuffle produces same result with same seed
    result1 = filter_instances(instances.copy(), filter_spec="", slice_spec="", shuffle=True)
    result2 = filter_instances(instances.copy(), filter_spec="", slice_spec="", shuffle=True)
    assert result1 == result2
    # Test that shuffled result is different from original order
    result_no_shuffle = filter_instances(instances.copy(), filter_spec="", slice_spec="", shuffle=False)
    assert result1 != result_no_shuffle


def test_filter_instances_empty_list():
    """Test filter_instances with empty input list"""
    result = filter_instances([], filter_spec=r".*", slice_spec="0:5", shuffle=True)
    assert result == []


def test_filter_instances_no_matches():
    """Test filter_instances when regex matches nothing"""
    instances = [{"instance_id": "django__test1"}, {"instance_id": "flask__test2"}]
    result = filter_instances(instances, filter_spec=r"nonexistent__.*", slice_spec="")
    assert result == []


def _empty_loc_entry(instance_id: str, model_name: str, patch: str) -> dict:
    return {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": patch,
        "localization": None,
        "localization_parse_error": None,
    }


def test_update_preds_file_new_file(tmp_path):
    """Test update_preds_file when output file doesn't exist"""
    output_path = tmp_path / "preds.json"
    update_preds_file(output_path, "test__instance__1", "test_model", "test_result")

    assert output_path.exists()
    result = json.loads(output_path.read_text())
    expected = {"test__instance__1": _empty_loc_entry("test__instance__1", "test_model", "test_result")}
    assert result == expected


def test_update_preds_file_with_localization(tmp_path):
    """Test update_preds_file records localization data in preds.json"""
    output_path = tmp_path / "preds.json"
    localization = Localization(
        root_cause=[{"file": "foo.py", "line": 10, "statement": "x = 1"}],
        reasoning="x is wrong",
        confidence="high",
    )
    update_preds_file(
        output_path,
        "test__instance__1",
        "test_model",
        "test_result",
        localization=localization,
    )

    result = json.loads(output_path.read_text())
    entry = result["test__instance__1"]
    assert entry["model_patch"] == "test_result"
    assert entry["localization"] == {
        "root_cause": [{"file": "foo.py", "line": 10, "statement": "x = 1"}],
        "reasoning": "x is wrong",
        "confidence": "high",
    }
    assert entry["localization_parse_error"] is None


def test_update_preds_file_with_parse_error(tmp_path):
    """Test update_preds_file records localization_parse_error when parsing fails"""
    output_path = tmp_path / "preds.json"
    update_preds_file(
        output_path,
        "test__instance__1",
        "test_model",
        "raw_submission",
        localization=None,
        localization_parse_error="Missing delimiter",
    )

    result = json.loads(output_path.read_text())
    entry = result["test__instance__1"]
    assert entry["localization"] is None
    assert entry["localization_parse_error"] == "Missing delimiter"


def test_update_preds_file_existing_file(tmp_path):
    """Test update_preds_file when output file already exists"""
    output_path = tmp_path / "preds.json"

    # Create initial file with one instance
    initial_data = {
        "existing__instance": _empty_loc_entry("existing__instance", "old_model", "old_result"),
    }
    output_path.write_text(json.dumps(initial_data))

    # Add new instance
    update_preds_file(output_path, "new__instance", "new_model", "new_result")

    result = json.loads(output_path.read_text())
    expected = {
        "existing__instance": _empty_loc_entry("existing__instance", "old_model", "old_result"),
        "new__instance": _empty_loc_entry("new__instance", "new_model", "new_result"),
    }
    assert result == expected


def test_update_preds_file_overwrite_existing(tmp_path):
    """Test update_preds_file overwrites existing instance"""
    output_path = tmp_path / "preds.json"

    # Create initial file
    initial_data = {"test__instance": _empty_loc_entry("test__instance", "old_model", "old_result")}
    output_path.write_text(json.dumps(initial_data))

    # Update existing instance
    update_preds_file(output_path, "test__instance", "new_model", "new_result")

    result = json.loads(output_path.read_text())
    expected = {"test__instance": _empty_loc_entry("test__instance", "new_model", "new_result")}
    assert result == expected


def test_remove_from_preds_file_existing(tmp_path):
    """Test remove_from_preds_file removes existing instance"""
    output_path = tmp_path / "preds.json"

    # Create file with multiple instances
    initial_data = {
        "instance1": {"model_name_or_path": "model1", "instance_id": "instance1", "model_patch": "result1"},
        "instance2": {"model_name_or_path": "model2", "instance_id": "instance2", "model_patch": "result2"},
    }
    output_path.write_text(json.dumps(initial_data))

    # Remove one instance
    remove_from_preds_file(output_path, "instance1")

    result = json.loads(output_path.read_text())
    expected = {"instance2": {"model_name_or_path": "model2", "instance_id": "instance2", "model_patch": "result2"}}
    assert result == expected


def test_remove_from_preds_file_nonexistent_instance(tmp_path):
    """Test remove_from_preds_file with nonexistent instance"""
    output_path = tmp_path / "preds.json"

    initial_data = {"instance1": {"model_name_or_path": "model1", "instance_id": "instance1", "model_patch": "result1"}}
    output_path.write_text(json.dumps(initial_data))

    # Try to remove nonexistent instance
    remove_from_preds_file(output_path, "nonexistent")

    # File should be unchanged
    result = json.loads(output_path.read_text())
    assert result == initial_data


def test_remove_from_preds_file_no_file(tmp_path):
    """Test remove_from_preds_file when file doesn't exist"""
    output_path = tmp_path / "preds.json"

    # Should not raise an error
    remove_from_preds_file(output_path, "any_instance")

    # File should still not exist
    assert not output_path.exists()


@pytest.mark.slow
def test_redo_existing_false_skips_existing(github_test_data, tmp_path):
    """Test that redo_existing=False skips instances that already have results"""
    model_responses = github_test_data["model_responses"]

    # Create existing preds.json with one instance
    preds_file = tmp_path / "preds.json"
    existing_data = {
        "swe-agent__test-repo-1": {
            "model_name_or_path": "previous_model",
            "instance_id": "swe-agent__test-repo-1",
            "model_patch": "previous_result",
        }
    }
    preds_file.write_text(json.dumps(existing_data))

    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.side_effect = lambda **kwargs: _make_model_from_fixture(model_responses)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            workers=1,
            filter_spec="swe-agent__test-repo-1",
            redo_existing=False,
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
        )

    # Should still have the original result
    result = json.loads(preds_file.read_text())
    assert result == existing_data


@pytest.mark.slow
def test_redo_existing_true_overwrites_existing(github_test_data, tmp_path, container_executable):
    """Test that redo_existing=True processes instances even if they already have results"""
    model_responses = github_test_data["model_responses"]

    # Create existing preds.json with one instance
    preds_file = tmp_path / "preds.json"
    existing_data = {
        "swe-agent__test-repo-1": {
            "model_name_or_path": "previous_model",
            "instance_id": "swe-agent__test-repo-1",
            "model_patch": "previous_result",
        }
    }
    preds_file.write_text(json.dumps(existing_data))

    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.side_effect = lambda **kwargs: _make_model_from_fixture(model_responses, cost_per_call=0.1)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            workers=1,
            filter_spec="swe-agent__test-repo-1",
            redo_existing=True,
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
            environment_class="docker",
        )

    # Should have new result from deterministic model
    traj_file_path = package_dir.parent.parent / "tests" / "test_data" / "github_issue.traj.json"
    trajectory = json.loads(traj_file_path.read_text())
    _, _, expected_patch = trajectory[-1]["content"].partition("<<<MSWEA_PATCH_DELIMITER>>>")
    expected_patch = expected_patch.lstrip("\n")

    result = json.loads(preds_file.read_text())
    assert result["swe-agent__test-repo-1"]["model_patch"] == expected_patch
    assert result["swe-agent__test-repo-1"]["model_name_or_path"] == "deterministic"
    assert result["swe-agent__test-repo-1"]["localization"] is not None


class ExceptionModelConfig(BaseModel):
    model_name: str = "exception_model"


class ExceptionModel:
    """Test model that raises exceptions during processing."""

    def __init__(self, exception_type: type[Exception] = RuntimeError, exception_message: str = "Test exception"):
        self.exception_type = exception_type
        self.exception_message = exception_message
        self.cost = 0.0
        self.n_calls = 0
        self.config = ExceptionModelConfig()

    def query(self, *args, **kwargs):
        self.n_calls += 1
        raise self.exception_type(self.exception_message)

    def format_message(self, **kwargs) -> dict:
        return dict(**kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return [self.format_message(role="user", content=str(o)) for o in outputs]

    def get_template_vars(self, **kwargs) -> dict:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def serialize(self) -> dict:
        return {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_exception_handling_in_agent_run(tmp_path, workers, container_executable):
    """Test that exceptions during agent.run() are properly handled and recorded"""
    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = ExceptionModel(RuntimeError, "Agent processing failed")

        with patch("minisweagent.run.benchmarks.swebench.RunBatchProgressManager") as mock_progress_class:
            mock_progress_manager = mock_progress_class.return_value
            mock_progress_manager.render_group = None  # For Live context manager

            main(
                subset="_test",
                split="test",
                slice_spec="0:1",
                output=str(tmp_path),
                workers=workers,
                filter_spec="swe-agent__test-repo-1",
                config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
                environment_class="docker",
            )

    # Check that prediction file contains exception information
    preds_file = tmp_path / "preds.json"
    assert preds_file.exists()

    result = json.loads(preds_file.read_text())
    instance_id = "swe-agent__test-repo-1"
    assert instance_id in result
    assert result[instance_id]["model_patch"] == ""
    assert result[instance_id]["model_name_or_path"] == "exception_model"

    # Check that trajectory file contains exception information
    traj_file = tmp_path / instance_id / f"{instance_id}.traj.json"
    assert traj_file.exists()

    traj_data = json.loads(traj_file.read_text())
    assert traj_data["instance_id"] == instance_id
    assert traj_data["info"]["exit_status"] == "RuntimeError"
    assert traj_data["info"]["submission"] == ""
    assert traj_data["info"]["exception_str"] == "Agent processing failed"


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_different_exception_types(tmp_path, workers, container_executable):
    """Test that different exception types are properly recorded"""
    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = ExceptionModel(ValueError, "Invalid input provided")

        with patch("minisweagent.run.benchmarks.swebench.RunBatchProgressManager") as mock_progress_class:
            mock_progress_manager = mock_progress_class.return_value
            mock_progress_manager.render_group = None  # For Live context manager

            main(
                subset="_test",
                split="test",
                slice_spec="0:1",
                output=str(tmp_path),
                workers=workers,
                filter_spec="swe-agent__test-repo-1",
                config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
                environment_class="docker",
            )

    # Check trajectory file for correct exception type
    instance_id = "swe-agent__test-repo-1"
    traj_file = tmp_path / instance_id / f"{instance_id}.traj.json"
    traj_data = json.loads(traj_file.read_text())

    assert traj_data["info"]["exit_status"] == "ValueError"
    assert traj_data["info"]["submission"] == ""
    assert traj_data["info"]["exception_str"] == "Invalid input provided"


@pytest.mark.slow
def test_exception_handling_with_progress_manager(tmp_path, container_executable):
    """Test that progress manager receives exception notifications in multithreaded mode"""
    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = ExceptionModel(ConnectionError, "Network timeout")

        with patch("minisweagent.run.benchmarks.swebench.RunBatchProgressManager") as mock_progress_class:
            mock_progress_manager = mock_progress_class.return_value
            mock_progress_manager.render_group = None  # For Live context manager

            main(
                subset="_test",
                split="test",
                slice_spec="0:1",
                output=str(tmp_path),
                workers=2,  # Use multithreaded to test progress manager
                filter_spec="swe-agent__test-repo-1",
                config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
                environment_class="docker",
            )

            # Verify progress manager methods were called
            mock_progress_manager.on_instance_start.assert_called_once_with("swe-agent__test-repo-1")
            mock_progress_manager.on_instance_end.assert_called_once_with("swe-agent__test-repo-1", "ConnectionError")

            # on_uncaught_exception should not be called since exceptions are handled properly
            mock_progress_manager.on_uncaught_exception.assert_not_called()


class _RecordingEnv:
    """Fake environment that records executed commands. Commands containing `fail_on` return returncode 1."""

    def __init__(self, fail_on: str = ""):
        self.commands: list[str] = []
        self.fail_on = fail_on

    def execute(self, action: dict, cwd: str = "") -> dict:
        command = action["command"]
        self.commands.append(command)
        returncode = 1 if self.fail_on and self.fail_on in command else 0
        return {"output": "", "returncode": returncode, "exception_info": ""}


def test_resolve_pre_patch_path_not_configured():
    assert resolve_pre_patch_path({}, "some__instance-1") is None
    assert resolve_pre_patch_path({"run": {}}, "some__instance-1") is None


def test_resolve_pre_patch_path_from_dir(tmp_path):
    (tmp_path / "some__instance-1.patch").write_text("diff")
    (tmp_path / "some__instance-2.diff").write_text("diff")
    config = {"run": {"pre_patch_dir": str(tmp_path)}}
    assert resolve_pre_patch_path(config, "some__instance-1") == tmp_path / "some__instance-1.patch"
    assert resolve_pre_patch_path(config, "some__instance-2") == tmp_path / "some__instance-2.diff"


def test_resolve_pre_patch_path_from_dir_missing(tmp_path):
    config = {"run": {"pre_patch_dir": str(tmp_path)}}
    with pytest.raises(PrePatchMissingError, match="no__such-instance"):
        resolve_pre_patch_path(config, "no__such-instance")


def test_resolve_pre_patch_path_from_file_any_name(tmp_path):
    patch_path = tmp_path / "my-experiment-42.custom.patch"
    patch_path.write_text("diff")
    config = {"run": {"pre_patch_file": str(patch_path)}}
    assert resolve_pre_patch_path(config, "unrelated__instance-1") == patch_path


def test_resolve_pre_patch_path_from_file_missing(tmp_path):
    config = {"run": {"pre_patch_file": str(tmp_path / "nope.patch")}}
    with pytest.raises(PrePatchMissingError, match="nope.patch"):
        resolve_pre_patch_path(config, "some__instance-1")


def test_resolve_pre_patch_path_both_configured_raises(tmp_path):
    config = {"run": {"pre_patch_file": "a.patch", "pre_patch_dir": str(tmp_path)}}
    with pytest.raises(ValueError, match="only one"):
        resolve_pre_patch_path(config, "some__instance-1")


def test_apply_pre_patch_applies_and_commits(tmp_path):
    patch_content = "--- a/f.py\n+++ b/f.py\n@@\n-x = 'quotes' && $vars `backticks`\n+y\n"
    patch_path = tmp_path / "some__instance-1.patch"
    # write_bytes: the patch must reach the container byte-for-byte (no newline translation on Windows)
    patch_path.write_bytes(patch_content.encode())
    env = _RecordingEnv()

    apply_pre_patch(env, "some__instance-1", patch_path)

    assert len(env.commands) == 4
    write_cmd, apply_cmd, add_cmd, commit_cmd = env.commands
    # The patch is transferred base64-encoded and must decode back to the original content
    encoded = re.search(r"echo '([^']+)' \| base64 -d", write_cmd).group(1)
    assert base64.b64decode(encoded).decode() == patch_content
    assert apply_cmd.startswith("git apply")
    assert add_cmd == "git add -A"
    assert commit_cmd.startswith("git ") and "commit" in commit_cmd


def test_apply_pre_patch_failure_raises(tmp_path):
    patch_path = tmp_path / "some__instance-1.patch"
    patch_path.write_text("not a valid diff")
    env = _RecordingEnv(fail_on="git apply")

    with pytest.raises(PrePatchApplyError, match="git apply"):
        apply_pre_patch(env, "some__instance-1", patch_path)
    # Stops at the failing command: no add/commit afterwards
    assert len(env.commands) == 2
