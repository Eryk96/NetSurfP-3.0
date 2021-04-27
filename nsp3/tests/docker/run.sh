# run from root
sudo docker run --rm -it -v $(pwd)/nsp3/tests:/nsp3/tests nsp-docker \
 nsp3 predict \
 -c config.yml \
 -d model.pth \
 -i nsp3/tests/docker/dummy.txt

 sudo docker run --rm -it -v $(pwd)/nsp3/tests:/nsp3/tests nsp-docker \
 python /home/eryk/development/NSPThesis/biolib/predict.py

 sudo docker run --rm -it -v $(pwd)/biolib:/biolib nsp-docker python biolib/predict.py