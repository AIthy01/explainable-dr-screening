import subprocess


MODELS = [
    "vgg16",
    "resnet50",
    "densenet121",
    "efficientnet_b3"
]


for model in MODELS:

    print("=" * 60)
    print(f"Start training: {model}")
    print("=" * 60)


    result = subprocess.run(
        [
            "python",
            "train.py",
            "--model",
            model
        ]
    )


    if result.returncode != 0:
        print(f"{model} failed.")
        break


    print(f"{model} finished successfully.")


print("All experiments finished.")