SELECT * FROM `flex-power.playground_sales.historicaldata_yield_WEA_3`
ORDER BY Datetime

common table expression

SELECT 
  TIMESTAMP_TRUNC(
    DATETIME(TIMESTAMP(delivery_start__utc_), "Europe/Berlin"),
    HOUR
  ) AS hour,

  SUM(volume__mw_) AS volumeMW,
  AVG(price__unit_per_mwh_) AS dayaheadprice

FROM `flex-power.domain.bidding__auctions_market_results`

WHERE segment = 'DAY-AHEAD'
  AND counterparty = 'EPEX'
  AND granularity = 'HOURLY'
  AND delivery_area IN ('FIFTYHERTZ', 'TENNET')
  AND portfolio = 'TOTAL'

GROUP BY hour
ORDER BY hour DESC;





with combine_asset as (
  SELECT *,
  EXTRACT(HOUR FROM Datetime) AS houroftheday_asset,
  EXTRACT(MONTH FROM Datetime) AS month_asset,
  EXTRACT(DAY FROM Datetime) AS day_asset,
  GREATEST(Active_power_MWh, 0) AS Active_power_MWh_filled
  
FROM flex-power.playground_sales.df_combined
),

DAprice as (
  SELECT *,
  EXTRACT(HOUR FROM hour) AS houroftheday_p,
  EXTRACT(MONTH FROM hour) AS month_p,
  EXTRACT(DAY FROM hour) AS day_p

FROM flex-power.playground_sales.Dayaheadprice
WHERE hour >= '2024-01-01' AND hour < '2025-01-01'
),

RMVprice as (
  SELECT *
  FROM flex-power.playground_sales.df_RMV
)


SELECT
  t.Datetime,
  t.Active_power_MWh,
  p.*,
  r.*
FROM combine_asset t
LEFT JOIN DAprice p
ON t.day_asset = p.day_p AND t.month_asset = p.month_p AND t.houroftheday_asset = p.houroftheday_p
LEFT JOIN RMVprice r
  ON t.month_asset = r.Month
































SELECT * FROM `flex-power.domain.bidding__auctions_market_results_portfolios_incremental` 
WHERE $__timeFilter(delivery_start__utc_) AND segment = 'DAY-AHEAD' AND counterparty = 'EPEX' AND granularity = 'HOURLY'
LIMIT 50

SELECT * FROM `flex-power.domain.bidding__auctions_market_results` 
WHERE $__timeFilter(delivery_start__utc_)
ORDER BY delivery_start__utc_ ASC 



SELECT 
TIMESTAMP_TRUNC(delivery_start__utc_, HOUR) AS hour,
SUM (CASE
  WHEN direction = 'BUY' THEN -volume__mw_
  WHEN direction = 'SELL' THEN volume__mw_
  ELSE 0
END) AS volumeMWbuysell,
SUM (volume__mw_) AS volume_mw_sum,
AVG(price__unit_per_mwh_) AS dayaheadprice_avg,
# delivery_start__utc_, delivery_end__utc_, price__unit_per_mwh_, price_unit, volume__mw_, direction, segment, granularity
FROM `flex-power.domain.bidding__auctions_market_results` 
# `flex-power.domain.bidding__auctions_market_results_portfolios_incremental` 
WHERE $__timeFilter(delivery_start__utc_) AND segment = 'DAY-AHEAD' AND counterparty = 'EPEX' AND granularity = 'HOURLY' AND delivery_area IN ('FIFTYHERTZ', 'TENNET') AND portfolio = 'TOTAL'
GROUP BY hour
ORDER BY hour DESC;












SELECT  delivery_start__utc_, delivery_end__utc_, price__unit_per_mwh_, price_unit, volume__mw_, direction, segment
FROM `flex-power.domain.bidding__auctions_market_results_portfolios_incremental` 
# `flex-power.domain.bidding__auctions_market_results`
WHERE $__timeFilter(delivery_start__utc_) AND segment = 'DAY-AHEAD'
LIMIT 50 





SELECT
  TIMESTAMP_TRUNC(delivery_start__utc_, HOUR) AS hour,
  SUM(CASE
    WHEN direction = 'BUY' THEN -volume__mw_ * price__unit_per_mwh_
    WHEN direction = 'SELL' THEN volume__mw_ * price__unit_per_mwh_
    ELSE 0
  END) AS pnl
FROM flex-power.domain.bidding__auctions_market_results_portfolios_incremental
WHERE $__timeFilter(delivery_start__utc_)
  AND segment = 'DAY-AHEAD'
  AND counterparty = 'EPEX'
  AND granularity = 'HOURLY'
GROUP BY hour
ORDER BY hour DESC;

# SELECT TIMESTAMP_TRUNC(delivery_start__utc_, DAY) AS day,


SELECT * FROM flex-power.domain.bidding__auctions_market_results_portfolios_incremental WHERE $__timeFilter(delivery_start__utc_)


SELECT column_name
FROM `flex-power.domain.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'bidding__auctions_market_results_portfolios_incremental';

SELECT DISTINCT delivery_area
FROM flex-power.domain.bidding__auctions_market_results_portfolios_incremental;



import sqlite3  # Import the sqlite3 library

def connect_db():
    """Connect to the SQLite database."""
    return sqlite3.connect("trades.sqlite")

def list_tables():
    """List all tables in the database."""
    connection = connect_db()
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]  # Extract table names
    connection.close()
    return tables

def list_columns(table_name):
    """List all column names from a specific table."""
    connection = connect_db()
    cursor = connection.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")  # Get table info
    columns = [column[1] for column in cursor.fetchall()]  # Extract column names
    connection.close()
    return columns


# output print column name
tables = list_tables()
print("Tables in the database:", tables)

# print columns name
if tables:
    for table in tables:
        columns = list_columns(table)
        print(f"\n📌 Columns in table '{table}': {columns}")






# SQL
import sqlite3

CREATE TABLE Student (
    ROLL_NO INT PRIMARY KEY,
    NAME VARCHAR(50),
    ADDRESS VARCHAR(100),
    PHONE VARCHAR(15),
    HEIGHT REAL,
    AGE INT
);


SELECT x, y
FROM x 
LIMIT x
WHERE
GROUP BY xx
HAVING # group by and having are a group
ORDER BY score DESC; #order by Age / ACS / DESC



# select distinct (choosing unique value) specific to Java and Javascipt course
SELECT DISTINCT score, course
from geeksforgeeks 
WHERE course IN ('Java','JavaScript');


SELECT COUNT(DISTINCT ReleaseYear) 
FROM FilmLocations 
WHERE ProductionCompany = "Warner Bros. Pictures";


# inserting new row/data
INSERT INTO table_name (column1, column2, column3) 
VALUES ( value1, value2, value), 
(value x, value y, value z);


INSERT INTO table_name (column1, column2, ... )
VALUES (value1, value2, ... )
;


# inserting new column from other table
INSERT INTO first_table(names_of_columns1) 
SELECT names_of_columns2 
FROM second_table; 

# Copy Specific Rows and Insert 
INSERT INTO table1 
SELECT * FROM table2 
WHERE condition; 
# WHERE x BETWEEN 12 AND 20;
# WHERE x IS NULL;
# WHERE X = 'abc';
# WHERE <> 'abc';           # not equal

# WHERE xcolumn IN ('Abc', 15)
# WHERE NOT city = 'Bandung' AND city = 'Bali'


# calculating something and make a new column
SELECT X,
Y,
Price * Quantity AS new_column_pricetimesquantity
from xyz

# Aggregate functions statistics: AVG(), COUNT(), MIN(), MAX(), SUM()
SELECT AVG(Price) AS newcolumn_average_price
SELECT COUNT (*) AS newcolumn_count     # count only counting for non NULL value
SELECT COUNT(DISTINCT customerID) FROM xyz      # not counting duplicate data

# Wildcards: searching a specific word in the middle or first/last sentences
LIKE '%Pizza'
'Pizza%'
'%Pizza%'
's%@gmail.com'

# for a single letter wildcard, use underscore
WHERE columnname LIKE '_pizza'


# GROUP BY and HAVING (combination, cannot use WHERE with GROUP BY)
GROUP BY CustomerID
HAVING SUM(Salary) >= 5000 AND AVG(Salary) > 5000;


# Update the column NAME and set the value
# modifying column name etc
UPDATE Customer # table name
SET 
CustomerName = 'abc', 
Country = 'USA' 
WHERE CustomerID = 1;


SELECT 
  Extended_Step,
  Job_Code,
  Pay_Type
FROM 
  salary_range_by_job_classification
WHERE 
  Union_Code = '990' AND SetID IN ('SFMTA', 'COMMN');


# delete data or specific row
DELETE FROM GFG_Employees  # table name
WHERE department = 'Development';


# alter
# ALTER TABLE table_name
# [ADD | DROP | MODIFY] column_name datatype;

ALTER TABLE table_name
MODIFY COLUMN column_name datatype; #modifying column datatype

ALTER TABLE table_name
RENAME COLUMN old_name TO new_name;

ALTER TABLE table_name
RENAME TO new_table_name;

ALTER TABLE `PETSALE` CHANGE `PET` `ANIMAL` varchar(20);        # rename a column


# drop or truncate data
DROP DATABASE student_data;  # drop the data permanently
TRUNCATE TABLE Student_details; # truncate table, delete all the rows data without erasing the table itself








# joining two table: Albums and Tracks
SELECT 
    a.AlbumId, #a. or t. is an alias for each table
    a.Title,
    COUNT(t.TrackId) AS TrackCount
FROM 
    Albums a
JOIN 
    Tracks t ON a.AlbumId = t.AlbumId
GROUP BY 
    a.AlbumId, a.Title
HAVING 
    COUNT(t.TrackId) >= 12;

# inner join
INNER JOIN # irisan




#1
SELECT *
FROM Tracks
WHERE Milliseconds >= 5000000

#2
SELECT *
FROM Invoices
WHERE Total BETWEEN 5 AND 15

#3
SELECT *
FROM Customers
WHERE State IN ('RJ','DF','AB','BC','CA','WA','NY')

#4
SELECT *
FROM Invoices
WHERE CustomerId IN (56, 58)
  AND Total BETWEEN 1 AND 5;

#5
SELECT *
FROM Tracks
WHERE Name like 'All%'

#6
SELECT *
FROM Customers
WHERE Email like 'J%@gmail.com'

#7
SELECT *
FROM Invoices
WHERE BillingCity IN ('Brasília', 'Edmonton', 'Vancouver')
ORDER BY InvoiceId DESC

#8
SELECT 
    CustomerId,
    COUNT(InvoiceId) AS NumberOfOrders
FROM 
    Invoices
GROUP BY 
    CustomerId
ORDER BY 
    NumberOfOrders DESC;

#9
SELECT 
    a.AlbumId,
    a.Title,
    COUNT(t.TrackId) AS TrackCount
FROM 
    Albums a
JOIN 
    Tracks t ON a.AlbumId = t.AlbumId
GROUP BY 
    a.AlbumId, a.Title
HAVING 
    COUNT(t.TrackId) >= 12;






# example on joining multiple tables
SELECT 
    albums.Title AS AlbumTitle,
    tracks.UnitPrice
FROM 
    artists
JOIN 
    albums ON artists.ArtistId = albums.ArtistId
JOIN 
    tracks ON albums.AlbumId = tracks.AlbumId
WHERE 
    artists.Name = 'Audioslave';

# EXAMPLE 2
SELECT 
    customers.FirstName,
    customers.LastName
FROM 
    customers
LEFT JOIN 
    invoices ON customers.CustomerId = invoices.CustomerId
WHERE 
    invoices.InvoiceId IS NULL;


# EXAMPLE 3
SELECT 
    tracks.Name, tracks.UnitPrice,
    SUM(tracks.UnitPrice) AS summy
FROM 
    albums
JOIN 
    tracks ON albums.AlbumId = tracks.AlbumId
WHERE 
    albums.Title = 'Big Ones';

# EXAMPLE 4
SELECT 
    artists.Name AS Artist,
    COUNT(albums.AlbumId) AS AlbumCount
FROM 
    artists
JOIN 
    albums ON artists.ArtistId = albums.ArtistId
WHERE 
    artists.Name = 'Led Zeppelin'
GROUP BY 
    artists.Name;



# example 5
SELECT 
    c.FirstName || ' ' || c.LastName AS FullName,
    c.City,
    c.Email,
    COUNT(i.InvoiceId) AS TotalInvoices
FROM 
    Customers c
JOIN 
    Invoices i ON c.CustomerId = i.CustomerId
GROUP BY 
    c.CustomerId, c.FirstName, c.LastName, c.City, c.Email;


# example 6
SELECT 
    t.Name AS TrackName,
    a.Title AS AlbumTitle,
    ar.ArtistId,
    t.TrackId
FROM 
    Tracks t
JOIN 
    Albums a ON t.AlbumId = a.AlbumId
JOIN 
    Artists ar ON a.ArtistId = ar.ArtistId;


# example 7
SELECT 
    c.CustomerId,
    c.FirstName,
    c.LastName,
    c.City AS CustomerCity,
    i.BillingCity
FROM 
    Customers c
JOIN 
    Invoices i ON c.CustomerId = i.CustomerId
WHERE 
    c.City <> i.BillingCity;










# showing data for specific date/ hour
import sqlite3
import pandas as pd

# Connect or create
conn = sqlite3.connect("mydata.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS measurements (
    timestamp TEXT,
    value REAL
)
""")

# Insert example data
sample_data = [
    ("2023-01-01 00:00:00", 10.5),
    ("2023-01-01 00:15:00", 12.2),
    ("2023-01-01 00:30:00", 11.7),
    ("2023-01-02 12:00:00", 13.5),
    ("2023-01-02 12:15:00", 13.9),
    ("2023-01-03 08:30:00", 14.2)
]
cursor.executemany("INSERT INTO measurements VALUES (?, ?)", sample_data)
conn.commit()

# for specific date
specific_date = '2023-01-02'

query = """
SELECT *
FROM measurements
WHERE DATE(timestamp) = ?
"""
# using panda
df = pd.read_sql_query(query, conn, params=(specific_date,))
print(df)

# using SQL
cursor.execute(query,(specific_date))





# for a range of data
start = '2023-01-02 12:00:00'
end = '2023-01-02 12:30:00'

query = """
SELECT *
FROM measurements
WHERE timestamp BETWEEN ? AND ?
"""

# using panda
df = pd.read_sql_query(query, conn, params=(start, end))
print(df)

# using SQL
# Execute query
cursor.execute(query, (start, end))
rows = cursor.fetchall()
for row in rows:
    print(row)
# Close connection
conn.close()




# -- (Double hyphen) for single-line comments:
# /* This is a 
#   multi-line comment */

# If you are working with Python and SQLite3, you might see 
# triple quotes (""" """ or ''' ''').




# using SUBSTR to split a word
# SUBSTR(string, start, length)
SELECT
  FirstName,
  LastName,
  LOWER(SUBSTR(FirstName, 1, 4) || SUBSTR(LastName, 1, 2)) AS EmployeeUserID
FROM employees;


# Show a list of employees who have worked for the company for 15 or more years using the current date function. Sort by lastname ascending.
SELECT 
  FirstName,
  LastName,
  HireDate,
  CAST((JULIANDAY(CURRENT_DATE) - JULIANDAY(HireDate)) / 365 AS INTEGER) AS YearsWorked
FROM 
  employees
WHERE 
  (JULIANDAY(CURRENT_DATE) - JULIANDAY(HireDate)) / 365 >= 15
ORDER BY 
  LastName ASC;




# counting null in a column
SELECT 
  SUM(CASE WHEN FirstName IS NULL THEN 1 ELSE 0 END) AS FirstName_NULLs,
  SUM(CASE WHEN LastName IS NULL THEN 1 ELSE 0 END) AS LastName_NULLs,
  SUM(CASE WHEN Company IS NULL THEN 1 ELSE 0 END) AS Company_NULLs,
  SUM(CASE WHEN Address IS NULL THEN 1 ELSE 0 END) AS Address_NULLs,
  SUM(CASE WHEN City IS NULL THEN 1 ELSE 0 END) AS City_NULLs,
  SUM(CASE WHEN State IS NULL THEN 1 ELSE 0 END) AS State_NULLs,
  SUM(CASE WHEN Country IS NULL THEN 1 ELSE 0 END) AS Country_NULLs,
  SUM(CASE WHEN PostalCode IS NULL THEN 1 ELSE 0 END) AS PostalCode_NULLs,
  SUM(CASE WHEN Phone IS NULL THEN 1 ELSE 0 END) AS Phone_NULLs,
  SUM(CASE WHEN Email IS NULL THEN 1 ELSE 0 END) AS Email_NULLs
FROM Customers;

# combining several columns
SELECT 
  FirstName,
  LastName,
  InvoiceId,
  InvoiceId || '-' || FirstName || '-' || LastName AS NewCustomerInvoiceID
FROM 
  Customers
JOIN
  Invoices ON Customers.CustomerId = Invoices.CustomerId
ORDER BY 
  FirstName, LastName, InvoiceId;












# showing all tables name and its columns name
import sqlite3

# Connect to database
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()

# example of checking table is there or not
# check if table exists
print('Check if STUDENT table exists in the database:')
listOfTables = cur.execute(
  """SELECT tableName FROM sqlite_master WHERE type='table'
  AND tableName='STUDENT'; """).fetchall()

if listOfTables == []:
    print('Table not found!')
else:
    print('Table found!')

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# cursor.execute("""UPDATE STAFF SET NAME = 'Ram', AGE = 30,  
# DEPARTMENT = 'Biology' WHERE DEPARTMENT = 'Computer';""")
tables = cursor.fetchall()

# Print tables and their columns
for table_name in tables:
    table = table_name[0]
    print(f"Table: {table}")
    
    cursor.execute(f"PRAGMA table_info({table})")   
    columns = cursor.fetchall()
    
    for col in columns:
        print(f"  Column: {col[1]} ({col[2]})")  # name and type

# Index	Meaning
# [0]	Column ID
# [1]	Column name
# [2]	Column data type
# [3]	Not null? (0 or 1)
# [4]	Default value
# [5]	Primary key? (0 or 1)




# example 
# Import module 
import sqlite3 

# Connecting to sqlite 
conn = sqlite3.connect('geek.db') 

# Creating a cursor object using the 
# cursor() method 
cursor = conn.cursor() 

# erase the table if exists
# cursor.execute("DROP TABLE IF EXISTS STUDENT")

# Creating table 
table ="""CREATE TABLE STUDENT(NAME VARCHAR(255), CLASS VARCHAR(255), 
SECTION VARCHAR(255));"""
cursor.execute(table) 

# check the database creation data
if cursor:
    print("Database Created Successfully !")
else:
    print("Database Creation Failed !")

# Queries to INSERT records. 
cursor.execute( 
'''INSERT INTO STUDENT (CLASS, SECTION, NAME) VALUES ('7th', 'A', 'Raju')''') 

cursor.execute( 
'''INSERT INTO STUDENT (SECTION, NAME, CLASS) VALUES ('B', 'Shyam', '8th')''') 

cursor.execute( 
'''INSERT INTO STUDENT (NAME, CLASS, SECTION ) VALUES ('Baburao', '9th', 'C')''') 

# Display data inserted 
print("Data Inserted in the table: ") 
data = cursor.execute("""SELECT * FROM STUDENT WHERE Department = 'IT'""") 
for row in data: 
	print(row) 


# example fetchmany
print("Data Inserted in the table:")
cursor.execute("SELECT * FROM STUDENT WHERE Department = 'IT'")
rows = cursor.fetchmany(5)  # fetches up to 5 rows
for row in rows:
    print(row)

# Commit your changes in 
# the database	 
conn.commit() 

# Closing the connection 
conn.close()



# cursor.execute("""SELECT * from STUDENT WHERE First_name LIKE 'R%' """)  # first name starts with R: wildcard
# WHERE Student_ID = 777

# output = cursor.fetchmany(7) # limit for 7 data
# for row in output = 
#     print(row)

# output = cursor.fetchone() # retrieve data from the table and fetch only one record
# print(output)

# cursor.execute("""SELECT * from STUDENT WHERE First_name LIKE 'R%' """)  # first name starts with R: wildcard
# print (cursor.fetchall())






# cursor.fetchall()
# Purpose: Fetches all remaining rows of a query result.
# Returns: A list of tuples.
cursor.execute("SELECT * FROM test")
rows = cursor.fetchall()
print(rows)  # Output: [(1, 'Alice'), (2, 'Bob')]


# cursor.fetchmany(size)
cursor.fetchmany(size)
cursor.execute("SELECT * FROM test")
rows = cursor.fetchmany(1)
print(rows)  # Output: [(1, 'Alice')]







# establishing a connection to the database 
connection   = sqlite3.connect("sales.db") 
# creating a cursor object 
cursor = connection.cursor() 

# count of all the rows of the database 
count = "select count(*) from sales1"

cursor.execute(count) 

print("The count of all rows of the table  :") 
print(cursor.fetchone()[0])     # if not using [0], it is a tuple with one value
# cursor.fetchone() → (42,)  # a tuple with one value

# Closing database connection 
connection.close()






# To import a CSV file into an SQLite database
import sqlite3
import pandas as pd

df = pd.read_csv("your_file.csv")  # Replace with actual file name
conn = sqlite3.connect("your_database.db")  # Creates or connects to DB


df.to_sql("your_table_name", conn, if_exists="replace", index=False)
# "your_table_name": name of the SQL table you want to create.

cursor = conn.cursor()
cursor.execute("SELECT * FROM your_table_name LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()



# Select all Spanish customers that starts with either "G" or "R"
SELECT * FROM Customers
WHERE Country = 'Spain' AND (CustomerName LIKE 'G%' OR CustomerName LIKE 'R%')

# Select all customers that either:
# are from Spain and starts with either "G", or
# starts with the letter "R":
SELECT * FROM Customers
WHERE Country = 'Spain' AND CustomerName LIKE 'G%' OR CustomerName LIKE 'R%';



import sqlite3
# Connect to SQLite database (it will be created if it doesn't exist)
connection = sqlite3.connect("energy_data.sqlite")
cursor = connection.cursor()

# SQL statement to create the table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS energy_metrics (
        date TEXT,  -- ISO format: YYYY-MM-DD hh:mm:ss
        wind_offshore_mwh REAL,
        wind_onshore_mwh REAL,
        photovoltaics_mwh REAL,
        actual_grid_load_mwh REAL,
        residual_load_mwh REAL,
        hour_sin REAL,
        hour_cos REAL,
        month_sin REAL,
        month_cos REAL,
        dow_sin REAL,
        dow_cos REAL,
        lag_1 REAL,
        lag_4 REAL,
        rolling_std_96 REAL
    )
""")
