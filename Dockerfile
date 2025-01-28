
# Use the official python 3.7 image
FROM python:3.7

# setting the working directory to code 
WORKDIR /code

#copy the content of current directory into code
COPY ./requirements.txt /code/requirements.txt

#Install the requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt


#Setup a new user named "used"
RUN useradd user

# switch to the user "user"

USER user

# Set home to the user's home directory

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

#Setup the working directory to the user's home directory
WORKDIR $HOME/app

# copy the current directory contents to $HOME/app setting the owner to user
COPY --chown=user . $HOME/app

#start tht fast api on port 7860
CMD ["uvicorn" , "app:app" , "--host" , "0.0.0.0" , "--port" , "7860"]


