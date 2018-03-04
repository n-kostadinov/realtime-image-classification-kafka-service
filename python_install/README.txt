1 ) To install the python environment follow the instructions on: 
http://blog.kaggle.com/2016/02/05/how-to-get-started-with-data-science-in-containers/

2) IMPORTANT: Instead "sudo docker pull kaggle/python" you may want to run 
sudo docker build -t "kaggleandkafka/python:dockerfile" .
in this directory in order to additionally install kafka-python client

3) NOTE: The whole installation might require up to 10 GB of free space. If you are
running docker for the first time on your machine, make sure you have configured the working directory,
in the Kaggle example above it is /tmp/working 

4) NOTE: When following the kaggle tutorial and exending your .bashrc, you might prefer to add this line instead:
alias kpython='sudo docker run -v $PWD:/tmp/working -w=/tmp/working --rm -it kaggle/python python "$@"'
after that you can run the kafka service like this:
kpython Real_Time_Classification_Kafka_Service.py

5) To save you a ton of troubles, use Ubuntu 16.04.4 LTS
