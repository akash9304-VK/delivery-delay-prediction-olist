
import getpass
from sqlalchemy import URL, create_engine, text

password = getpass.getpass("Enter MySQL password: ")

url = URL.create(
    "mysql+pymysql",
    username="root",
    password=password,
    host="localhost",
    port=3306,
    database="delivery_delay",
)

engine = create_engine(url)

try:
    with engine.connect() as conn:
        print("Connection successful!")
        print("Database:", conn.execute(
            text("SELECT DATABASE()")
        ).scalar())
finally:
    engine.dispose()
