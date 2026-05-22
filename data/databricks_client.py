from databricks import sql
from config import (
    DATABRICKS_HOST,
    DATABRICKS_HTTP_PATH,
    DATABRICKS_TOKEN
)

def get_connection():
    return sql.connect(
        server_hostname = DATABRICKS_HOST,
        http_path       = DATABRICKS_HTTP_PATH,
        access_token    = DATABRICKS_TOKEN
    )