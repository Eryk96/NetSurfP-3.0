# Archive model
FROM pytorch/torchserve:0.3.0-cpu AS archive-model

COPY . .

RUN torch-model-archiver --model-name nsp3 \
    --version 1.0 \
    --serialized-file ./export.pt \
    --extra-files ./index_to_name.json, ./handler.py \
    --handler ./main.py  \
    --export-path ./

# Production
FROM pytorch/torchserve:0.3.0-cpu

COPY config.properties /home/model-server/config.properties
COPY --from=archive-model nsp3.mar /home/model-server/model-store