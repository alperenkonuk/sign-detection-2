FROM signimg

COPY . /home/test
WORKDIR /home/test

CMD python3 test.py
# CMD python3 test2.py
# CMD python3 detect.py --source inference/images/test5.jpg --weights weights/best.pt --conf 0.4 --save-txt