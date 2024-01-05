import torch
import cv2
from PIL import Image
from os import path, makedirs
from mimetypes import guess_type
from tqdm import tqdm, trange
from uuid import uuid4
from json import load, dump

import embedding_models.ResNet50_Embedding as ie


def process(session, input):
    try:
        model = ie.ResNet50_ImageEmbedder()

        file_list = []
        for i in input:
            file_list += _validate_source(i)

        if not path.exists(f"./.tmp/{session}/v/vectors"):
            makedirs(f"./.tmp/{session}/v/vectors")

        model.eval()
        with torch.no_grad():
            for file in tqdm(desc="Indexing video vector database", iterable=file_list):
                # Create video feed from file
                video = cv2.VideoCapture(file)
                _, frame = video.read()
                id = uuid4().hex
                # Save id and file path in tmp session folder in json file
                _save_id_to_file(session, id, path.abspath(file))
                for _ in trange(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
                    # Transform images into tensors
                    t = ie.get_transforms()(Image.fromarray(frame))
                    # Create embedding vector
                    vector = torch.squeeze(model(t.unsqueeze(0)))
                    # Save vector in tmp session folder
                    if not path.exists(f"./.tmp/{session}/v/vectors/{id}"):
                        makedirs(f"./.tmp/{session}/v/vectors/{id}")
                    torch.save(
                        vector, f"./.tmp/{session}/v/vectors/{id}/{uuid4().hex}.pt"
                    )
                    # Get next frame
                    _, frame = video.read()

                video.release()

    except Exception as e:
        raise e


def _validate_source(source):
    try:
        if guess_type(source)[0].startswith("video/"):
            return [source]
    except:
        pass

    return []


def _save_id_to_file(session, id, filename):
    # check if embedding map file exists, if not create it
    if not path.exists(f"./.tmp/{session}/v/embedding_map.json"):
        with open(f"./.tmp/{session}/v/embedding_map.json", "w") as f:
            dump({}, f)

    with open(f"./.tmp/{session}/v/embedding_map.json", "r") as j:
        data = load(j)
        data[id] = filename
        with open(f"./.tmp/{session}/v/embedding_map.json", "w") as f:
            dump(data, f)
