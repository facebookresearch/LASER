## Docker

An image docker has been created to help you with the settings of an environment here are the step to follow :

* Open a command prompt on the root of your LASER project
* Execute the command `docker build --tag=laser .`
* Once the image is built run `docker run -it laser`

A REST server on top of the embed task is under construction, 
to run it you'll have to expose the port 8099 by executing the next line instead of the last command.

* `docker run -p 80:81 -it laser`

Once you'll be inside of the docker container, you'll be able to run the Flask server by executing this command

*  `python app.py`