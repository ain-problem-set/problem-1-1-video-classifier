import numpy as np

import config as C
from benchmark import run as run_benchmark
from export import export_production
from predict import Predictor, collect_videos
from train_eval import run_evaluation


class FakeEncoder:
    def embed(self, frames, batch_size=32):
        v = np.zeros((len(frames), 8), np.float32)
        v[:, 0] = 1.0
        return v


def predictor(fake_artifacts):
    run_evaluation(fake_artifacts, n_repeats=1, n_boot=10)
    export_production(fake_artifacts)
    return Predictor(fake_artifacts / "models", encoder=FakeEncoder())


def test_predict_returns_routing_decision(fake_artifacts, make_video):
    r = predictor(fake_artifacts).predict(make_video(n=100))
    assert r["class"] in C.CLASSES
    assert 0.0 <= r["confidence"] <= 1.0
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-4
    expected = C.FALLBACK_CHUNKER if r["fallback"] else C.CHUNKERS[r["class"]]
    assert r["recommended_chunker"] == expected
    assert r["latency_ms"]["total"] > 0


def test_corrupt_file_reports_error_without_stopping_batch(fake_artifacts, make_video, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("nope")
    results = predictor(fake_artifacts).predict_many([bad, make_video(n=100)])
    assert "error" in results[0] and "class" in results[1]


def test_collect_videos_expands_directories(tmp_path, make_video):
    make_video("a.avi")
    make_video("b.avi")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in collect_videos([tmp_path])] == ["a.avi", "b.avi"]


def test_benchmark_reports_batch1_and_batch32(fake_artifacts, make_video):
    out = run_benchmark(predictor(fake_artifacts), [make_video(n=100)], repeats=1)
    assert out["batch1_ms"]["p50"] > 0 and out["batch32_throughput_vps"] > 0
