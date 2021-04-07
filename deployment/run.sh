docker run --rm -it \
-v $(pwd)/data:/data \
direct-model \
exec nsp3 predict -d data/data.txt