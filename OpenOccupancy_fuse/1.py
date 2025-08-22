import argparse
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
import torch

def main():
    parser = argparse.ArgumentParser(description="Inspect detection annotations presence")
    parser.add_argument("config", help="path to config file")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    try:
        if hasattr(cfg, "_cfg_dict") and "datetime" in cfg._cfg_dict:
            cfg._cfg_dict.pop("datetime")
    except Exception:
        pass

    dataset_cfg = cfg.data[args.split]
    dataset = build_dataset(dataset_cfg)

    missing_raw = 0
    missing_after_pipeline = 0
    examples = []

    for idx in range(min(len(dataset), args.max_samples)):
        ann = dataset.get_ann_info(idx)
        has_bbox_raw = ann.get("gt_bboxes_3d", None) is not None
        has_label_raw = ann.get("gt_labels_3d", None) is not None
        if not (has_bbox_raw and has_label_raw):
            missing_raw += 1

        # run through the pipeline up to PrepareDetAnnotations manually
        data_info = dataset.prepare_train_data(idx) if not dataset.test_mode else dataset.prepare_test_data(idx)
        if data_info is None:
            missing_after_pipeline += 1
            examples.append((idx, "pipeline returned None"))
            continue
        has_bbox_post = data_info.get("gt_bboxes_3d", None) is not None
        has_label_post = data_info.get("gt_labels_3d", None) is not None
        if not (has_bbox_post and has_label_post):
            missing_after_pipeline += 1
            examples.append((idx, f"post pipeline bbox={has_bbox_post}, label={has_label_post}"))

    print("Raw missing:", missing_raw)
    print("After pipeline missing:", missing_after_pipeline)
    print("Sample issues:", examples[:30])

if __name__ == "__main__":
    main()
