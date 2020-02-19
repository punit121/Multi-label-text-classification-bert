import argparse
from pathlib import Path
from tqdm import tqdm
import requests
import urllib3

PRETRAINED_VOCAB_FILES_MAP = {
    # ROBERTA
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-vocab.json",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json"
}

PRETRAINED_VOCAB_MERGES_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-merges.txt",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-merges.txt",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
}

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    # ROBERTA
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin"

}

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {

    # ROBERTA
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-config.json",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-config.json",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-config.json",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-config.json"

}


def http_get(url, target):
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    with open(target, "wb") as target_file:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                target_file.write(chunk)
    progress.close()


def download_pretrained_files(model_name, location):
    model_type = model_name.split("-")[0]
    print("model name is {}".format(model_name))
    location = location / model_name
    print("location is {}".format(location))
    location.mkdir(exist_ok=True)
    # download vocab files
    try:
        file_path = PRETRAINED_VOCAB_FILES_MAP[model_name]
        print("file path is {}".format(file_path))
        if model_type == "bert":
            file_name = "vocab.txt"
        if model_type == "distilbert":
            file_name = "vocab.txt"
        elif model_type == "xlnet":
            file_name = "spiece.model"
        elif model_type == "roberta":
            file_name = "vocab.json"

        target_path = location / file_name
        http_get(file_path, target_path)

    except:
        print(
            "error downloading model vocab {} for  model {}".format(
                file_path, model_name
            )
        )

    # download vocab merge file for Roberta
    if model_type == "roberta":
        try:
            file_path = PRETRAINED_VOCAB_MERGES_MAP[model_name]
            print(file_path)
            file_name = "merges.txt"
            target_path = location / file_name
            http_get(file_path, target_path)

        except:
            print("error downloading model merge file for {}".format(model_name))

    # download model files
    try:
        file_path = BERT_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]
        print(file_path)
        file_name = "pytorch_model.bin"
        target_path = location / file_name
        http_get(file_path, target_path)

    except:
        print("error downloading model file for {}".format(model_name))

    # download config files
    try:
        file_path = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name]
        print(file_path)
        file_name = "config.json"
        target_path = location / file_name
        http_get(file_path, target_path)

    except:
        print("error downloading model config for {}".format(model_name))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--location_dir",
        default=None,
        type=str,
        required=True,
        help="The location where pretrained model needs to be stored",
    )

    parser.add_argument(
        "--models",
        default=None,
        type=str,
        required=True,
        nargs="*",
        help="download the pretrained models",
    )

    args = parser.parse_args()
    print(args)
    Path(args.location_dir).mkdir(exist_ok=True)

    models = args.models
    #    [download_pretrained_files(k, location=Path(args.location_dir))
    #     for k, v in BERT_PRETRAINED_MODEL_ARCHIVE_MAP.items()]
    [
        download_pretrained_files(item, location=Path(args.location_dir))
        for item in args.models
    ]


if __name__ == "__main__":
    main()
