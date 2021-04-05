from handler import RequestHandler

service = RequestHandler()

def handle(data, context):
    if not service.initialized:
        service.initialize(context)

    if data is None:
        return None

    data = service.preprocess(data)
    data = service.inference(data)
    data = service.postprocess(data)

    return data
