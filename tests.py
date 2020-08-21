import os
import sys

from analysis.bins import BinApplier
from notebook_utils.constants import NO_HORROR
from notebook_utils.map_to_roget import RogetMapper

sys.path.append(os.getcwd())

from analysis.thresholds import ThresholdApplier
from notebook_utils.feature_reduction import FeatureReducer
from loading_utils.models import ModelLoader


def test_fr():
    ta = ThresholdApplier("pos")

    no_reduc_accs = []
    best_idx, accs, weights = ta.apply_thresholds(0, None, None, None, None)
    fr = FeatureReducer(model_loader=ta.model_loader)
    exh, rw, _ = fr.reduce_features(weights[best_idx], accs[best_idx])
    fr.plot_exhausted(exh, marker_size=20)


def test_fr_by_genre():
    ta = ThresholdApplier(model="pos", by_genre=True)

    no_reduc_accs = []
    best_idx, accs, weights, results = ta.apply_thresholds_by_genre(add_to_acc=no_reduc_accs)
    fr = FeatureReducer(model_loader=ta.model_loader)
    exh, rw = fr.reduce_features_by_genre(model_weights=weights, og_acc=accs)
    fr.plot_exhausted_by_genre(exh, marker_size=20)


def test_ml():
    loader = ModelLoader("clausal", by_genre=True)
    loader.load_model()


def test_ta_by_genre():
    ta = ThresholdApplier(model="wordnet", by_genre=True, load=False)
    accs, weights, preds, results = ta.apply_thresholds_by_genre()
    print(results)


def test_ta():
    ta = ThresholdApplier(model="wordnet", load=False)
    best_idx, accs, weights, results = ta.apply_thresholds(0, None, None, None, None)
    print(results)


def test_bins_by_genre():
    bbg = BinApplier(model="wordnet", by_genre=True)
    results = bbg.apply_bins()
    print(results["Adventure"]["Max Accuracy"])


def test_bins():
    bbg = BinApplier(model="wordnet")
    results = bbg.apply_bins()
    print(results["Adventure"]["Max Accuracy"])


def test_map_to_roget_full():
    ta = ThresholdApplier(model="wordnet", by_genre=True)
    best_idx, accs, weights, results = ta.apply_thresholds_by_genre()
    fr = FeatureReducer(model_loader=ta.model_loader)
    exh, rw = fr.reduce_features_by_genre(model_weights=weights, og_acc=accs)

    rm = RogetMapper(model_loader=fr.model_loader)
    wn_set = {genre: rm.model_loader.data.loc[rm.model_loader.data["@Genre"] == genre] for genre in NO_HORROR}
    NumsAndOutcomes = {genre: rm.model_loader.data.loc[rm.model_loader.data["@Genre"] == genre][["Book #", "@Genre", "@Outcome"]].reset_index(drop=True) for genre in NO_HORROR}

    wn_to_rocat = rm._map_to_roget_by_genre(wn_set, "Word", "Category", NumsAndOutcomes)
    wn_to_rocat_no_scale, wn_to_rocat_scaled = rm._concat_map_to_roget_by_genre(wn_to_rocat, "Category", NumsAndOutcomes)
    wn_rosect_no_scale, wn_rosect_scaled = rm._map_to_roget_by_genre(wn_to_rocat_no_scale, "Category", "Section", NumsAndOutcomes)

    full_wn_rosect_acc, full_wn_rosect_weights, wn_rosect_acc, wn_rosect_weights = rm.test_map_to_roget(wn_rosect_no_scale, map_to="Section")
    wn_rosect_df = rm.scale_mapped(wn_rosect_no_scale, )
    wn_rosect_exh, wn_rosect_rw = rm.reduce_features_by_genre(full_wn_rosect_weights, og_acc=full_wn_rosect_acc, data=wn_rosect_df)
    wn_to_rosect_reduced_acc = rm.plot_exhausted_by_genre(wn_rosect_exh, marker_size=20)


def test_map_to_roget():
    wn_rm = RogetMapper(model="wordnet", by_genre=True, jupyter=True)
    print(wn_rm.model_loader.data)


def test_regression():
    wn = FeatureReducer(model="wordnet", by_genre=True, estimator="lr")
    wn_accs, wn_weights, wn_preds = wn.predict_by_genre(disp_weights=False)
    print(wn_accs)


if __name__ == "__main__":
    # test_ta()
    test_bins_by_genre()
