import ibm_db
import sqlite3
import mysql.connectopr

# Database Credentials

dsn_hostname = "YourDb2Hostname"
dsn_uid = "db2admin"        
dsn_pwd = "Debby21212"      
dsn_driver = "{IBM DB2 ODBC DRIVER}"
dsn_database = "BLUDB"           
dsn_port = "32733"               
dsn_protocol = "TCPIP"           
dsn_security = "SSL"     

# Creating DSN Connection

dsn = (
    "DRIVER={0};"
    "DATABASE={1};"
    "HOSTNAME={2};"
    "PORT={3};"
    "PROTOCOL={4};"
    "UID={5};"
    "PWD={6};"
    "SECURITY={7};").format(dsn_driver, dsn_database, dsn_hostname, dsn_port, dsn_protocol, dsn_uid, dsn_pwd,dsn_security)

# Creating Database Connection

try:
    conn = ibm_db.connect(dsn, "", "")
    print ("Connected to database: ", dsn_database, "as user: ", dsn_uid, "on host: ", dsn_hostname)

except:
    print ("Unable to connect: ", ibm_db.conn_errormsg() )

# connection = sqlite3.connect('ibm9_db')
# cursor = connection.cursor()