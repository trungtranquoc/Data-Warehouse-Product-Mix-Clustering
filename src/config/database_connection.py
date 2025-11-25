from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

server_name = '127.0.0.1'
database_name = 'master'
username = 'sa'
password = 'trung_password123'
driver_name = 'ODBC Driver 18 for SQL Server'
# Add your port number here (1433 is default)
port_number = 1433

raw_connection_string = (
    f'DRIVER={{{driver_name}}};'
    f'SERVER={server_name},{port_number};' # Note the 'host,port' format
    f'DATABASE={database_name};'
    f'UID={username};'
    f'PWD={password};'
    # **CRITICAL: This is the trust connection setting you need**
    f'TrustServerCertificate=yes;' 
)

quoted_connection_string = quote_plus(raw_connection_string)

connection_url = f"mssql+pyodbc:///?odbc_connect={quoted_connection_string}"

mssql_engine = create_engine(connection_url)

try:
    with mssql_engine.connect() as conn:
        # optional: issue a lightweight query
        conn.execute(text("SELECT 1"))
    print("Connecting to SQL Server Database successful")
except SQLAlchemyError as e:
    print("Connection failed to Database SQL Server. Please check the connection parameters !")
    raise e
