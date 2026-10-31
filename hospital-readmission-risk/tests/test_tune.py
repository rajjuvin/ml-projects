import json

import pytest

from readmission.data import DATA_FILE, split_xy
from readmission.models import DEFAULT_PARAMS, DISPLAY_NAMES, build_model, load_params, model_params
from readmission.tune import FAST_N_ITER, main, search_space, tune_model


def test_every_model_has_a_search_space_that_contains_its_defaults(frame):
    _, y, _ = split_xy(frame)
    for name in DISPLAY_NAMES:
        space = search_space(name, y)
        assert set(DEFAULT_PARAMS[name]) <= set(space)
        for key, value in DEFAULT_PARAMS[name].items():
            if key != "scale_pos_weight":
                assert value in space[key], (name, key)


def test_xgboost_search_includes_the_class_imbalance_ratio(frame):
    _, y, _ = split_xy(frame)
    ratio = (y == 0).sum() / (y == 1).sum()
    assert ratio in search_space("xgboost", y)["scale_pos_weight"]


def test_unknown_model_has_no_search_space(frame):
    _, y, _ = split_xy(frame)
    with pytest.raises(ValueError, match="Unknown model"):
        search_space("not_a_model", y)


def test_params_override_defaults_but_keep_the_rest():
    tuned = model_params("decision_tree", {"max_depth": 3})
    assert tuned == {"max_depth": 3, "min_samples_leaf": DEFAULT_PARAMS["decision_tree"]["min_samples_leaf"]}
    estimator = build_model("decision_tree", params={"max_depth": 3}).named_steps["model"]
    assert estimator.max_depth == 3
    assert estimator.min_samples_leaf == 50


def test_load_params_reads_the_file_and_restores_tuples(in_tmp_cwd, write_params_file):
    path = in_tmp_cwd / "best.json"
    write_params_file(path, {"mlp": {"hidden_layer_sizes": [128, 64], "alpha": 1.0}})
    assert load_params(path) == {"mlp": {"hidden_layer_sizes": (128, 64), "alpha": 1.0}}


def test_load_params_without_a_file(in_tmp_cwd):
    assert load_params() == {}  # no default file: built-in defaults apply
    with pytest.raises(FileNotFoundError):
        load_params(in_tmp_cwd / "missing.json")


def test_tune_model_returns_parameters_from_the_search_space(frame):
    X, y, groups = split_xy(frame)
    result = tune_model("decision_tree", X, y, groups, n_jobs=1, n_iter=FAST_N_ITER)
    space = search_space("decision_tree", y)
    assert result["search"] == f"randomized (n_iter={FAST_N_ITER})"
    assert set(result["params"]) == set(space)
    assert all(value in space[key] for key, value in result["params"].items())
    assert 0 < result["cv_pr_auc_mean"] < 1


def test_tune_model_searches_the_whole_grid_when_it_is_small(frame):
    X, y, groups = split_xy(frame)
    result = tune_model("random_forest", X, y, groups, n_jobs=1)
    assert result["search"] == "grid"
    assert result["n_candidates"] == 2


def test_tune_cli_writes_params_and_keeps_other_models(in_tmp_cwd, raw_frame):
    raw_frame.to_csv(in_tmp_cwd / DATA_FILE, index=False)
    out = in_tmp_cwd / "params" / "best.json"
    base = ["--data-dir", str(in_tmp_cwd), "--fast", "--n-jobs", "1", "--out", str(out)]
    main([*base, "--models", "decision_tree"])
    main([*base, "--models", "logistic_regression"])

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert list(saved["models"]) == ["logistic_regression", "decision_tree"]  # decision_tree survived the 2nd run
    entry = saved["models"]["decision_tree"]
    assert entry["fast_run"] is True
    assert len(entry["data_file_sha256"]) == 64
    assert "max_depth" in load_params(out)["decision_tree"]


def test_fast_tuning_refuses_to_overwrite_the_default_file(in_tmp_cwd):
    with pytest.raises(SystemExit):
        main(["--fast", "--models", "decision_tree"])
    assert not (in_tmp_cwd / "reports").exists()
