import argparse
import torch

from yolov3_pytorch.engine.inferencer import Inferencer

WEIGHTS     = "./results/pretrained_models/YOLOv3_Tiny-COCO-20231107.pth.tar"
CPU_WEIGHTS = "./results/pretrained_models/YOLOv3_Tiny-COCO-20231107-cpu.pth.tar"
MODEL_CFG   = "./model_configs/COCO-Detection/yolov3_tiny.cfg"
IMG_SIZE    = 416


def save_cpu_weights() -> None:
    # map_location="cpu" moves all tensors to CPU at load time — no model needed
    ckpt = torch.load(WEIGHTS, map_location="cpu")

    for key, value in ckpt.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.dtype}")
        elif isinstance(value, dict):
            print(f"  {key}: <dict with {len(value)} entries>")

    torch.save(ckpt, CPU_WEIGHTS)
    print(f"CPU weights saved to: {CPU_WEIGHTS}")


def main() -> None:
    save_cpu_weights()

    opts = argparse.Namespace(
        inputs="data/examples/dog.jpg",
        output="./results/inference/",
        class_names_path="./data/coco.names",
        model_config_path=MODEL_CFG,
        img_size=IMG_SIZE,
        gray=False,
        weights=CPU_WEIGHTS,
        half=False,
        fuse=False,
        show_image=False,
        save_txt=False,
        fourcc="mp4v",
        conf_thresh=0.25,
        iou_thresh=0.45,
        augment=False,
        filter_classes=None,
        agnostic_nms=False,
        device="cpu",
    )

    app = Inferencer(opts)
    app.inference()


if __name__ == "__main__":
    main()
