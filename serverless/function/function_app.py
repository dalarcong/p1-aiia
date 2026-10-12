import logging
import azure.functions as func

app = func.FunctionApp()

@app.function_name(name="ProcessFile")
@app.blob_trigger(
    arg_name="archivo",
    path="documents/in/{name}",
    connection="DocumentsStorage",
    source="EventGrid",
)
def procesar_archivo(archivo: func.InputStream):
    logging.info(
        "File received: %s | Size: %s bytes",
        archivo.name,
        archivo.length,
    )
