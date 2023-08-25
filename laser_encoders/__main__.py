import argparse

from laser_encoders.download_models import download_models


def main():
    parser = argparse.ArgumentParser(description="LASER: Download Laser models")
    parser.add_argument(
        "command", choices=["download_models"], help="Command to execute"
    )
    parser.add_argument(
        "--laser",
        type=str,
        help="Laser model to download",
    )
    parser.add_argument(
        "--lang",
        type=str,
        help="The language name in FLORES200 format",
    )
    parser.add_argument(
        "--spm", action="store_false", help="Do not download the SPM model?"
    )
    parser.add_argument(
        "--model-dir", type=str, help="The directory to download the models to"
    )
    args = parser.parse_args()

    if args.command == "download_models":
        download_models(args)


if __name__ == "__main__":
    main()
