## LASER Docker Image

This image provides a convenient way to run LASER in a Docker container.
To build the image, run the following command from the root of the LASER directory:

```
docker build --tag=laser docker
```
Once the image is built, you can run it with the following command:

```
docker run -it laser
```
**Note:** If you want to expose a local port to the REST server on top of the embed task, you can do so by executing the following command instead of the last command:

```
docker run -it -p [CHANGEME_LOCAL_PORT]:80 laser python app.py
```
This will override the command line entrypoint of the Docker container.

Example:

```
docker run -it -p 8081:80 laser python app.py
```

This Flask server will serve a REST Api that can be use by calling your server with this URL :

```
http://127.0.0.1:[CHANGEME_LOCAL_PORT]/vectorize?q=[YOUR_SENTENCE_URL_ENCODED]&lang=[LANGUAGE]
```

Example:

```
http://127.0.0.1:8081/vectorize?q=ki%20lo%20'orukọ%20ẹ&lang=yor
```

Sample response:
```
{
    "content": "ki lo 'orukọ ẹ",
    "embedding": [
        [
            -0.10241681337356567,
            0.11120740324258804,
            -0.26641348004341125,
            -0.055699944496154785,
            ....
            ....
            ....
            -0.034048307687044144,
            0.11005636304616928,
            -0.3238321840763092,
            -0.060631975531578064,
            -0.19269055128097534,
        ]
}
```

Here is an example of how you can send requests to it with python:

```python
import requests
import numpy as np
url = "http://127.0.0.1:[CHANGEME_LOCAL_PORT]/vectorize"
params = {"q": "Hey, how are you?\nI'm OK and you?", "lang": "en"}
resp = requests.get(url=url, params=params).json()
print(resp["embedding"])
```