import torch
import timm

from src.models.classifier import create_model, ENCODER_ZOO
from src.label_map import NUM_CLASSES


def run_experiments():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Timm Version: {timm.__version__}\n")

    dummy_input = torch.randn(2, 3, 224, 224)

    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Available encoders: {list(ENCODER_ZOO.keys())}\n")

    for encoder_name in ENCODER_ZOO.keys():
        print("=" * 60)
        print(f"Testing encoder: {encoder_name}")
        print(f"Timm model name: {ENCODER_ZOO[encoder_name]}")

        try:
            aggregation = "cls_token" if "vit" in encoder_name else "gap"

            model = create_model(
                encoder_name=encoder_name,
                aggregation=aggregation,
                num_classes=NUM_CLASSES,
                pretrained=False,
            )

            total_params = sum(
                p.numel() for p in model.parameters()
                if p.requires_grad
            )

            print(f"Trainable parameters: {total_params:,}")

            model.eval()

            with torch.no_grad():
                outputs = model(dummy_input)

            print(
                f"Output shape: {list(outputs.shape)} "
                f"(Expected: [2, {NUM_CLASSES}])"
            )

            assert outputs.shape == (2, NUM_CLASSES)

            print("Status: SUCCESS")

        except Exception as e:
            print("Status: FAILED")
            print(f"Error: {e}")

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    run_experiments()