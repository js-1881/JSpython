import sqlite3    # Import the sqlite3 library

def connect_db():
    """"connect to the database"""
    return sqlite3.connect("trades.sqlite")


# Task 1.1a: Compute total buy volume
def compute_total_buy_volume ():
    """computes total buy volume"""

    connection = connect_db() #connect to the database using previous def
    cursor = connection.cursor()

    # Execute query that calculates the sum of the 'quantity' column for all trades 
    # where the 'side' is 'buy'
    cursor.execute("""SELECT SUM(quantity) FROM epex_12_20_12_13 WHERE side = 'buy'""")
    

    # Fetch the result of the query, take the first element [0]
    result = cursor.fetchone()[0]

    connection.close() #close database'

    return result if result else 0 # Return 0 if there are no 'buy' trades


# to call the function
buy_volume = compute_total_buy_volume()
print(buy_volume)




# import sqlite3 USING PANDA
import pandas as pd

def compute_total_buy_volume_pandas():
    """Computes total buy volume using pandas"""

    # Connect to the SQLite database
    connection = sqlite3.connect("trades.sqlite")

    # Read the table into a DataFrame
    df = pd.read_sql_query("SELECT * FROM epex_12_20_12_13", connection)

    # Filter where side is 'buy' and sum the 'quantity' column
    total_buy_volume = df[df['side'] == 'buy']['quantity'].sum()

    # Close the connection
    connection.close()

    return total_buy_volume




# both filtering side = buy and strategy = strategy 1 using panda
import sqlite3
import pandas as pd

def compute_total_buy_volume_strategy_1():
    """Computes total buy volume where side = 'buy' and strategy = 'strategy_1' using pandas"""

    # Connect to the SQLite database
    connection = sqlite3.connect("trades.sqlite")

    # Load the entire table into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM epex_12_20_12_13", connection)

    # Filter where side is 'buy' AND strategy is 'strategy_1'
    filtered_df = df[(df['side'] == 'buy') & (df['strategy'] == 'strategy_1')]

    # Sum the quantity column
    total_volume = filtered_df['quantity'].sum()




    # EXAMPLE OF USING PANDA GROUP TO SHOW ALL THE RESULTS
    # Group and aggregate
    grouped = df.groupby(['side', 'strategy'])['quantity'].sum().reset_index()

    # Sort by side and strategy
    grouped = grouped.sort_values(by=['side', 'strategy'])

     # Print the results
    print("Total volume for each side and strategy:")
    for _, row in grouped.iterrows():
        print(f"Side: {row['side']}, Strategy: {row['strategy']}, Total Volume: {row['quantity']}")

    for index, row in grouped.iterrows():
        print(index, row['side'], row['strategy'], row['quantity'])

        # example to print tidier
        # print(f"{index} side = {row['side']} strategy = {row['strategy']} total = {row['quantity']}")

    # Close the database connection
    connection.close()

    return total_volume

    





# using panda using PARAMETER side and strategy
def compute_total_volume(side, strategy):
    """Computes total trade volume filtered by side and strategy using pandas"""

    connection = sqlite3.connect("trades.sqlite")
    df = pd.read_sql_query("SELECT * FROM epex_12_20_12_13", connection)

    filtered_df = df[(df['side'] == side) & (df['strategy'] == strategy)]
    # filtered_df = df[(df['side'] == 'buy') | (df['side'] == 'sell')]        for OR |
    # filtered_df = df[~(df['strategy'] == 'strategy_1')]       for NOT ~
    
    total_volume = filtered_df['quantity'].sum()

    connection.close()
    return total_volume

side = 'buy'
strategy = 'strategy_1'
total = compute_total_volume(side, strategy)
print(f"side = {side}, strategy = {strategy}, total = {total}")





# using SQL side ? and strategy ?
import sqlite3

def compute_total_volume_sql(side, strategy):
    """Computes total trade volume using SQL with filters on side and strategy"""

    # Connect to the SQLite database
    connection = sqlite3.connect("trades.sqlite")
    cursor = connection.cursor()

    # SQL query with placeholders to prevent SQL injection
    query = """
    SELECT SUM(quantity) 
    FROM epex_12_20_12_13 
    WHERE side = ? AND strategy = ?
    """

    # Execute the query with parameters
    cursor.execute(query, (side, strategy))

    # Fetch the result
    result = cursor.fetchone()[0]

    # Close the connection
    connection.close()


    # Return the result or 0 if there are no matches
    return result if result else 0

total = compute_total_volume_sql('buy', 'strategy_1')
side = 'buy'
strategy = 'strategy_1'
print(f"side = {side}, strategy = {strategy}, total = {total}")



# using sql to print all the side and strategy looping
def print_volume_by_side_and_strategy_sql():
    """Prints total volume grouped by side and strategy using SQL only"""

    connection = sqlite3.connect("trades.sqlite")
    cursor = connection.cursor()

    # SQL query to group and sum
    # order is important
    query = """
    SELECT side, strategy, SUM(quantity) as total_volume  
    FROM epex_12_20_12_13
    GROUP BY side, strategy
    ORDER BY side, strategy
    """

    cursor.execute(query)
    results = cursor.fetchall()
    connection.close()

    # Print the results
    print("Total volume for each side and strategy:")
    for side, strategy, volume in results:
        print(f"Side: {side}, Strategy: {strategy}, Total Volume: {volume}")











#################################

# Task 1.1b: Compute total sell volume
def compute_total_sell_volume ():
    """computes total sell volume"""

    connection = connect_db() #connect to the database using previous def
    cursor = connection.cursor()

    # Execute query that calculates the sum of the 'quantity' column for all trades 
    # where the 'side' is 'sell'
    cursor.execute("""SELECT SUM(quantity) FROM epex_12_20_12_13 WHERE side = 'sell'""")
    

    # Fetch the result of the query, take the first element [0]
    result = cursor.fetchone()[0]

    connection.close() #close database'

    return result if result else 0 # Return 0 if there are no 'buy' trades


# print the calculated volume
print("Total Sell Volume:", compute_total_sell_volume())






# Task 1.2 computing the profit/loss
def compute_pnl(strategy_id: str)-> float:
    """computes profit and loss of each strategy"""
    connection = connect_db()
    cursor = connection.cursor()

    #calculating total profit/loss based on buy/sell transaction
    #when sell, quantity * price
    #when buy, -quantity * price
    cursor.execute("""
        SELECT SUM(
            CASE
                WHEN side = 'sell' THEN quantity * price
                WHEN side = 'buy' THEN -quantity * price
                ELSE 0  
            END  
        ) FROM epex_12_20_12_13 WHERE strategy = ? """,
        (strategy_id,))
            
    result = cursor.fetchone()[0]
    connection.close()

    return result if result else 0


# Fetch all unique strategy IDs from the database
def get_all_strategies():
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("SELECT DISTINCT strategy FROM epex_12_20_12_13")

    # Extracting strategy names inside the table
    strategies = [row[0] for row in cursor.fetchall()]

    connection.close()
    return strategies


# Compute and print PnL for all strategies
    # calling all the listed strategies
strategies = get_all_strategies()

# Loop through each strategy in the list and compute its PnL
for strategy in strategies:
    pnl = compute_pnl(strategy)  # Compute profit/loss for the current strategy

    print(f"{strategy}: {pnl:3f} EUR") # Print the strategy name and its profit/loss


# task 1.2 without using function
import sqlite3

# Connect to the database
connection = sqlite3.connect("trades.sqlite")
cursor = connection.cursor()

# Get all unique strategies
cursor.execute("SELECT DISTINCT strategy FROM epex_12_20_12_13")
strategies = [row[0] for row in cursor.fetchall()]

# Loop through each strategy and compute PnL
for strategy123 in strategies:
    cursor.execute("""
        SELECT SUM(
            CASE
                WHEN side = 'sell' THEN quantity * price
                WHEN side = 'buy' THEN -quantity * price
                ELSE 0
            END
        )
        FROM epex_12_20_12_13
        WHERE strategy = ?
    """, (strategy123,))
    
    result = cursor.fetchone()[0]
    pnl = result if result else 0

    print(f"{strategy123}: {pnl:.2f} EUR")

# Close the connection
connection.close()



# task 1.2 with panda
def compute_pnl_pandas():
    """Computes profit and loss per strategy using pandas"""

    connection = sqlite3.connect("trades.sqlite")
    df = pd.read_sql_query("SELECT strategy, side, quantity, price FROM epex_12_20_12_13", connection)
    connection.close()

    # Calculate PnL column
    df['pnl'] = df.apply(
        lambda row: row['quantity'] * row['price'] if row['side'] == 'sell' else -row['quantity'] * row['price'],
        axis=1
    )

    print(df[df['side'] == 'sell'][['pnl']].head()) # To show only PnL column for sells:

    # Group by strategy and sum PnL
    grouped_pnl = df.groupby('strategy')['pnl'].sum()

    # Print result
    for strategy, pnl in grouped_pnl.items():
        print(f"{strategy}: {pnl:.2f} EUR")



# Task 1.3 

from flask import Flask, jsonify # import flask for creating a web API, convert to json
from datetime import datetime # import datetime to generate timestamps


# Create a flask web app
app = Flask(__name__)

# Defines a URL path and links it to a function to expose the profit/loss computation
@app.route("/pnl/<strategy_id>", methods=["GET"])

def get_pnl(strategy_id):
    """API endpoint to get the profit/loss of a strategy"""

    # compute the amount for a given strategy
    pnl_amount = compute_pnl(strategy_id)

    # create a JSON response
    response = {
        "strategy": strategy_id,
        "value": pnl_amount,
        "unit": "euro",
        "capture time": datetime.utcnow().isoformat() #timestamp
    }

    return jsonify(response) #return JSON response to the user


# run the flask application if the script is executed
if __name__ == "__main__":
    app.run(debug=True) # run API in debug mode
    