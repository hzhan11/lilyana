import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import launcher


class LauncherTests(unittest.TestCase):
    def test_every_lesson_has_both_local_start_scripts(self):
        for target in launcher.PROGRAMS:
            if not target.startswith("package"):
                folder = launcher.source_dir(target).parent
                for filename in ("start.bat", "start.sh"):
                    text = (folder / filename).read_text(encoding="utf-8")
                    self.assertIn(f"--lesson {target} ", text)

    def test_local_no_arguments_and_help_never_start_programs(self):
        for target in launcher.PROGRAMS:
            if target.startswith("package"):
                continue
            for args in ([], ["help"], ["--help"], ["-h"]):
                with self.subTest(target=target, args=args):
                    output = io.StringIO()
                    with patch.object(launcher.subprocess, "run") as run:
                        with contextlib.redirect_stdout(output):
                            if args in (["--help"], ["-h"]):
                                with self.assertRaises(SystemExit) as error:
                                    launcher.main(["--lesson", target, *args])
                                self.assertEqual(error.exception.code, 0)
                            else:
                                self.assertEqual(launcher.main(["--lesson", target, *args]), 0)
                        run.assert_not_called()
                    self.assertIn("usage:", output.getvalue())
                    for action in launcher.actions_for(target):
                        self.assertIn(action, output.getvalue())
                    for filename in launcher.PROGRAMS[target][1].values():
                        self.assertIn(filename, output.getvalue())

    def test_local_action_forwards_arguments_and_failure(self):
        with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as run:
            self.assertEqual(launcher.main(["--lesson", "13", "training", "--epochs", "2"]), 7)
        run.assert_called_once_with(
            [sys.executable, "train_model.py", "--epochs", "2"],
            cwd=launcher.source_dir("13"), check=False,
        )

    def test_local_invalid_action_fails_without_starting_a_program(self):
        with patch.object(launcher.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                launcher.main(["--lesson", "1", "training"])
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_target_aliases(self):
        for value in ("8", "class8", "class8_cnn_playground"):
            self.assertEqual(launcher.resolve_target(value), "8")
        self.assertEqual(launcher.resolve_target("package2"), "package2")

    def test_registered_files_exist(self):
        for target, (_, programs) in launcher.PROGRAMS.items():
            directory = launcher.source_dir(target)
            self.assertTrue((directory / "requirements.txt").is_file(), target)
            for filename in programs.values():
                self.assertTrue((directory / filename).is_file(), (target, filename))

    def test_root_help_does_not_install_anything(self):
        for args in ([], ["help"], ["--help"]):
            output = io.StringIO()
            with patch.object(launcher.subprocess, "run") as run:
                with contextlib.redirect_stdout(output):
                    if args == ["--help"]:
                        with self.assertRaises(SystemExit) as error:
                            launcher.main(args)
                        self.assertEqual(error.exception.code, 0)
                    else:
                        self.assertEqual(launcher.main(args), 0)
            run.assert_not_called()
            self.assertIn("install", output.getvalue())
            self.assertIn(".venv", output.getvalue())

    def test_training_alias_and_filename(self):
        expected = [["train_model.py"]]
        for action in ("train", "training", "train_model.py"):
            self.assertEqual(launcher.commands_for("13", action), expected)

    def test_cnn_training_uses_gui(self):
        self.assertEqual(launcher.commands_for("8", "training"), [["app.py"]])
        self.assertEqual(launcher.commands_for("7", "cnn"), [["train_cnn.py"]])

    def test_class15_training_forwards_dataset_and_options(self):
        for action in ("training", "train", "train.py"):
            with self.subTest(action=action):
                with patch.object(
                    launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0),
                ) as run:
                    self.assertEqual(
                        launcher.main(["--lesson", "15", action, "2", "--epochs", "1"]), 0,
                    )
                run.assert_called_once_with(
                    [sys.executable, "train.py", "2", "--epochs", "1"],
                    cwd=launcher.source_dir("15"), check=False,
                )

    def test_unit_tests_are_discovered_in_each_source_directory(self):
        for target in launcher.UNIT_TEST_TARGETS:
            command, = launcher.commands_for(target, "check" if target == "15" else "test")
            self.assertEqual(command[:3], ["-m", "unittest", "discover"])
            self.assertIn("test_*.py", command)

    def test_class15_model_evaluation_is_separate_from_code_checks(self):
        self.assertEqual(launcher.commands_for("15", "test"), [["test.py"]])
        with patch.object(
            launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            self.assertEqual(launcher.main(["--lesson", "15", "test", "2"]), 0)
        run.assert_called_once_with(
            [sys.executable, "test.py", "2"], cwd=launcher.source_dir("15"), check=False,
        )

    def test_class15_combined_training_and_evaluation(self):
        for action, args, filename in (
            ("mtrain", ["--epochs", "1"], "mtrain.py"),
            ("mtrain.py", ["--epochs", "1"], "mtrain.py"),
            ("test", ["m"], "test.py"),
            ("test.py", ["m"], "test.py"),
        ):
            with self.subTest(action=action):
                with patch.object(
                    launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0),
                ) as run:
                    self.assertEqual(launcher.main(["--lesson", "15", action, *args]), 0)
                run.assert_called_once_with(
                    [sys.executable, filename, *args], cwd=launcher.source_dir("15"), check=False,
                )

    def test_metrics_test_runs_all_three_programs(self):
        self.assertEqual(
            launcher.commands_for("6", "test"),
            [["test_acc.py"], ["test_metrics.py"], ["test_tfpn.py"]],
        )

    def test_install_is_only_available_at_repository_root(self):
        for target in launcher.PROGRAMS:
            self.assertNotIn("install", launcher.actions_for(target))
            with self.assertRaisesRegex(ValueError, "repository root"):
                launcher.commands_for(target, "install")

    def test_shared_requirements_include_every_lesson_and_package(self):
        path = launcher.ROOT / "src" / "requirements.txt"
        included = {
            (path.parent / line.removeprefix("-r ")).resolve()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("-r ")
        }
        expected = {
            (launcher.source_dir(target) / "requirements.txt").resolve()
            for target in launcher.PROGRAMS
        }
        self.assertEqual(included, expected)

    def test_arguments_interpreter_and_working_directory(self):
        with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(launcher.main(["--lesson", "13", "training", "--epochs", "2"]), 0)
        run.assert_called_once_with(
            [sys.executable, "train_model.py", "--epochs", "2"],
            cwd=launcher.source_dir("13"), check=False,
        )

    def test_child_failure_is_propagated_and_stops_remaining_programs(self):
        with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as run:
            self.assertEqual(launcher.main(["--lesson", "6", "test"]), 7)
        self.assertEqual(run.call_count, 1)

    def test_invalid_requests_fail_without_starting_a_program(self):
        for args in (
            ["99", "app"], ["--lesson"], ["--lesson", "99", "app"],
            ["--lesson", "3", "test"], ["--lesson", "3", "..\\main.py"],
            ["--lesson", "3", "install"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(launcher.subprocess, "run") as run, self.assertRaises(SystemExit) as error:
                    launcher.main(args)
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_can_start_from_another_directory(self):
        result = subprocess.run(
            [sys.executable, str(Path(launcher.__file__)), "--help"],
            cwd=launcher.ROOT.parent, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("shared repository .venv", result.stdout)


class SharedEnvironmentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="lilyana setup ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        root_patch = patch.object(launcher, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.requirements = self.root / "src" / "requirements.txt"
        self.requirements.parent.mkdir()
        self.requirements.write_text("# No external packages needed for this fixture.\n", encoding="utf-8")
        self.python = launcher.environment_python()

    def create_interpreter_marker(self):
        self.python.parent.mkdir(parents=True)
        self.python.touch()

    def test_create_environment_then_install_with_its_python(self):
        def execute(command, **kwargs):
            if command[1:3] == ["-m", "venv"]:
                self.create_interpreter_marker()
            return subprocess.CompletedProcess(command, 0)

        with patch.object(launcher.subprocess, "run", side_effect=execute) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(["install"]), 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].args[0],
            [sys.executable, "-m", "venv", str(self.root / ".venv")],
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            [str(self.python), "-m", "pip", "install", "-r", str(self.requirements)],
        )

    def test_reuses_existing_environment(self):
        self.create_interpreter_marker()
        with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.install_environment(), 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:4], [str(self.python), "-m", "pip", "install"])

    def test_creation_failure_stops_before_pip(self):
        with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.install_environment(), 7)
        self.assertEqual(run.call_count, 1)

    def test_missing_interpreter_after_creation_is_an_error(self):
        with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.install_environment(), 1)
        self.assertEqual(run.call_count, 1)

    def test_incomplete_environment_is_not_overwritten(self):
        (self.root / ".venv").mkdir()
        with patch.object(launcher.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.install_environment(), 1)
        run.assert_not_called()

    def test_missing_requirements_stops_before_creation(self):
        self.requirements.unlink()
        with patch.object(launcher.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.install_environment(), 1)
        run.assert_not_called()

    def test_pip_failure_is_not_reported_as_success(self):
        self.create_interpreter_marker()
        output = io.StringIO()
        with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 9)):
            with contextlib.redirect_stdout(output):
                self.assertEqual(launcher.install_environment(), 9)
        self.assertNotIn("Setup complete", output.getvalue())

    def test_real_offline_install_and_automatic_shell_environment_selection(self):
        project = Path(launcher.__file__).resolve().parent.parent
        for relative in (
            "start.bat", "start.sh", "src/launcher.py", "src/start.bat", "src/start.sh",
            "study/class13_house_price_regression/start.bat",
            "study/class13_house_price_regression/start.sh",
        ):
            destination = self.root.joinpath(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project.joinpath(*relative.split("/")), destination)
        lesson = self.root / "study" / "class13_house_price_regression"
        (lesson / "src").mkdir()
        (lesson / "src" / "train_model.py").write_text(
            "import json, os, sys\n"
            "print(json.dumps({'prefix': sys.prefix, 'cwd': os.getcwd(), 'args': sys.argv[1:]}))\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.pop("PYTHON", None)
        environment.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")

        def execute(command, cwd):
            result = subprocess.run(
                command, cwd=cwd, env=environment, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

        execute([sys.executable, str(self.root / "src" / "launcher.py"), "install"], project)
        self.assertTrue(self.python.is_file())
        shell_commands = []
        if os.name == "nt":
            shell_commands.append(([os.environ["COMSPEC"], "/d", "/c", "start.bat"], "start.bat"))
        shell = shutil.which("sh") or shutil.which("bash")
        if shell is None and os.name == "nt":
            git = shutil.which("git")
            if git:
                candidate = Path(git).parents[1] / "bin" / "bash.exe"
                if candidate.is_file():
                    shell = str(candidate)
        if shell:
            shell_commands.append(([shell, "start.sh"], "start.sh"))
        self.assertTrue(shell_commands, "A platform shell is required for launcher integration tests.")
        for command, name in shell_commands:
            with self.subTest(shell=name):
                self.assertIn("Setup complete", execute([*command, "install"], self.root))
                self.assertIn("Actions:", execute(command, lesson))
                result = json.loads(execute(
                    [*command, "training", "--output", "a path with spaces"], lesson,
                ))
                self.assertEqual(Path(result["prefix"]).resolve(), (self.root / ".venv").resolve())
                self.assertEqual(Path(result["cwd"]).resolve(), (lesson / "src").resolve())
                self.assertEqual(result["args"], ["--output", "a path with spaces"])


if __name__ == "__main__":
    unittest.main()
