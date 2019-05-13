## Docker

An image docker has been created to help you with the settings of an environment here are the step to follow :

* Open a command prompt on the root of your LASER project
* Execute the command `docker build --tag=laser docker`
* Once the image is built run `docker run -it laser`

A REST server on top of the embed task is under developement,
to run it you'll have to expose a local port [CHANGEME_LOCAL_PORT] by executing the next line instead of the last command. It'll overinde the command line entrypoint of your docker container.

* `docker run -p [CHANGEME_LOCAL_PORT]:80 -it laser python app.py`

This Flask server will serve a REST Api that can be use by calling your server with this URL :

*   http://127.0.0.1:[CHANGEME_LOCAL_PORT]/vectorize?q=[YOUR_SENTENCE_URL_ENCODED]&lang=[LANGUAGE]