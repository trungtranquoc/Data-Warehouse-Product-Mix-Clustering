from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

server_name = 'localhost' 
database_name = 'postgres'
username = 'postgres'
password = 'Trungtq'
driver_name = 'PostgreSQL JDBC Driver'

# Add your port number here (1433 is default)
port_number = 5432

# Modified connection_url with the port:
connection_url = f"postgresql+psycopg2://{username}:{password}@{server_name}:{port_number}/{database_name}"

postgre_engine = create_engine(connection_url)

try:
    with postgre_engine.connect() as conn:
        # optional: issue a lightweight query
        conn.execute(text("SELECT 1"))
    print("Connecting to PostGre Database successful")
except SQLAlchemyError as e:
    print("Connection failed to Database PostgreSQL. Please check the connection parameters !")
    raise e

