from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import anonymizer.preprocessing.upscale as upscale_module
from anonymizer.pipeline import PipelineResult
from cli import _collect_images, build_parser, main


def _write_jpg(path: Path, h: int = 100, w: int = 150) -> Path:
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _mock_result(regions: int = 0, tokens: int = 0) -> PipelineResult:
    return PipelineResult(
        output_path=Path("out.jpg"),
        tokens_found=tokens,
        regions_redacted=regions,
    )


class TestBuildParser:
    def test_returns_argument_parser(self):
        import argparse
        assert isinstance(build_parser(), argparse.ArgumentParser)

    def test_inputs_positional(self):
        args = build_parser().parse_args(["a.jpg", "-o", "out/"])
        assert args.inputs == [Path("a.jpg")]

    def test_multiple_inputs(self):
        args = build_parser().parse_args(["a.jpg", "b.jpg", "-o", "out/"])
        assert len(args.inputs) == 2

    def test_output_dir_defaults_to_output_folder(self):
        args = build_parser().parse_args(["a.jpg"])
        assert args.output_dir.name == "output"

    def test_no_upscale_default_false(self):
        args = build_parser().parse_args(["a.jpg", "-o", "out/"])
        assert args.no_upscale is False

    def test_no_upscale_flag_sets_true(self):
        args = build_parser().parse_args(["a.jpg", "-o", "out/", "--no-upscale"])
        assert args.no_upscale is True

    def test_quiet_default_false(self):
        args = build_parser().parse_args(["a.jpg", "-o", "out/"])
        assert args.quiet is False

    def test_quiet_flag_sets_true(self):
        args = build_parser().parse_args(["a.jpg", "-o", "out/", "--quiet"])
        assert args.quiet is True

    def test_output_dir_stored_as_path(self):
        args = build_parser().parse_args(["a.jpg", "-o", "results/"])
        assert isinstance(args.output_dir, Path)

    def test_workers_default_one(self):
        args = build_parser().parse_args(["a.jpg", "-o", "out/"])
        assert args.workers == 1

    def test_workers_flag_parsed(self):
        args = build_parser().parse_args(["a.jpg", "-o", "out/", "--workers", "4"])
        assert args.workers == 4


class TestCollectImages:
    def test_file_path_returned_as_is(self, tmp_path):
        f = _write_jpg(tmp_path / "a.jpg")
        assert _collect_images([f]) == [f]

    def test_directory_returns_images_inside(self, tmp_path):
        _write_jpg(tmp_path / "a.jpg")
        _write_jpg(tmp_path / "b.jpeg")
        found = _collect_images([tmp_path])
        assert len(found) == 2

    def test_directory_excludes_non_images(self, tmp_path):
        _write_jpg(tmp_path / "a.jpg")
        (tmp_path / "notes.txt").write_text("ignore me")
        found = _collect_images([tmp_path])
        assert all(f.suffix.lower() in {".jpg", ".jpeg"} for f in found)

    def test_multiple_dirs_combined(self, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_jpg(d1 / "a.jpg")
        _write_jpg(d2 / "b.jpg")
        found = _collect_images([d1, d2])
        assert len(found) == 2

    def test_mix_of_files_and_dirs(self, tmp_path):
        f = _write_jpg(tmp_path / "solo.jpg")
        d = tmp_path / "batch"
        d.mkdir()
        _write_jpg(d / "x.jpg")
        found = _collect_images([f, d])
        assert len(found) == 2

    def test_empty_directory_returns_empty(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert _collect_images([d]) == []


class TestMainReturnCode:
    def test_returns_zero_on_success(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        with patch("cli.run", return_value=_mock_result()):
            code = main([str(src), "-o", str(tmp_path / "out")])
        assert code == 0

    def test_returns_two_when_all_fail(self, tmp_path):
        missing = tmp_path / "ghost.jpg"
        code = main([str(missing), "-o", str(tmp_path / "out")])
        assert code == 2

    def test_partial_failure_returns_one(self, tmp_path):
        good = _write_jpg(tmp_path / "good.jpg")
        missing = tmp_path / "ghost.jpg"
        with patch("cli.run", side_effect=[_mock_result(), FileNotFoundError("no")]):
            code = main([str(good), str(missing), "-o", str(tmp_path / "out")])
        assert code == 1

    def test_all_success_multiple_files(self, tmp_path):
        a = _write_jpg(tmp_path / "a.jpg")
        b = _write_jpg(tmp_path / "b.jpg")
        with patch("cli.run", return_value=_mock_result()):
            code = main([str(a), str(b), "-o", str(tmp_path / "out")])
        assert code == 0

    def test_no_images_found_returns_two(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        code = main([str(d), "-o", str(tmp_path / "out")])
        assert code == 2

    def test_directory_input_processes_all_images(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _write_jpg(src_dir / "a.jpg")
        _write_jpg(src_dir / "b.jpg")
        mock_run = MagicMock(return_value=_mock_result())
        with patch("cli.run", mock_run):
            code = main([str(src_dir), "-o", str(tmp_path / "out")])
        assert code == 0
        assert mock_run.call_count == 2

    def test_workers_flag_processes_all_images(self, tmp_path):
        a = _write_jpg(tmp_path / "a.jpg")
        b = _write_jpg(tmp_path / "b.jpg")
        mock_run = MagicMock(return_value=_mock_result())
        with patch("cli.run", mock_run):
            code = main([str(a), str(b), "-o", str(tmp_path / "out"), "--workers", "2"])
        assert code == 0
        assert mock_run.call_count == 2


class TestOutputDirectory:
    def test_output_dir_created_if_missing(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        out_dir = tmp_path / "new" / "nested"
        with patch("cli.run", return_value=_mock_result()):
            main([str(src), "-o", str(out_dir)])
        assert out_dir.is_dir()

    def test_existing_output_dir_not_raised(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        with patch("cli.run", return_value=_mock_result()):
            code = main([str(src), "-o", str(out_dir)])
        assert code == 0


class TestNoUpscaleFlag:
    def setup_method(self):
        upscale_module.UPSCALE_ENABLED = True

    def teardown_method(self):
        upscale_module.UPSCALE_ENABLED = True

    def test_no_upscale_disables_module_flag(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        with patch("cli.run", return_value=_mock_result()):
            main([str(src), "-o", str(tmp_path / "out"), "--no-upscale"])
        assert upscale_module.UPSCALE_ENABLED is False

    def test_without_flag_leaves_module_flag_true(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        with patch("cli.run", return_value=_mock_result()):
            main([str(src), "-o", str(tmp_path / "out")])
        assert upscale_module.UPSCALE_ENABLED is True


class TestQuietFlag:
    def test_quiet_suppresses_output(self, tmp_path, capsys):
        src = _write_jpg(tmp_path / "in.jpg")
        with patch("cli.run", return_value=_mock_result(regions=1)):
            main([str(src), "-o", str(tmp_path / "out"), "--quiet"])
        assert capsys.readouterr().out == ""

    def test_without_quiet_prints_per_file_result(self, tmp_path, capsys):
        src = _write_jpg(tmp_path / "in.jpg")
        with patch("cli.run", return_value=_mock_result(regions=2)):
            main([str(src), "-o", str(tmp_path / "out")])
        out = capsys.readouterr().out
        assert "in.jpg" in out
        assert "2 region(s)" in out


class TestRunCallArguments:
    def test_run_called_with_correct_input_path(self, tmp_path):
        src = _write_jpg(tmp_path / "photo.jpg")
        calls = []
        def fake_run(inp, out):
            calls.append((inp, out))
            return _mock_result()
        with patch("cli.run", side_effect=fake_run):
            main([str(src), "-o", str(tmp_path / "out")])
        assert calls[0][0] == src

    def test_run_called_with_output_in_output_dir(self, tmp_path):
        src = _write_jpg(tmp_path / "photo.jpg")
        out_dir = tmp_path / "out"
        calls = []
        def fake_run(inp, out):
            calls.append((inp, out))
            return _mock_result()
        with patch("cli.run", side_effect=fake_run):
            main([str(src), "-o", str(out_dir)])
        assert calls[0][1] == out_dir / "photo.jpg"

    def test_run_called_once_per_input(self, tmp_path):
        a = _write_jpg(tmp_path / "a.jpg")
        b = _write_jpg(tmp_path / "b.jpg")
        mock_run = MagicMock(return_value=_mock_result())
        with patch("cli.run", mock_run):
            main([str(a), str(b), "-o", str(tmp_path / "out")])
        assert mock_run.call_count == 2
