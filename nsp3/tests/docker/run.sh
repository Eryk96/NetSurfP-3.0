# run from root
sudo docker run --rm -it -v $(pwd)/nsp3/tests:/nsp3/tests nsp-docker \
 nsp3 predict \
 -c config.yml \
 -d model.pth \
 -i nsp3/tests/deployment/dummy.txt