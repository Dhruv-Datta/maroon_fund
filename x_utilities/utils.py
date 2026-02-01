import pandas as pd
# This file is to allow for multi ticker/date support

def getInfo():
    try:
        filedata = pd.read_csv('x_utilities/utils.csv')
    except pd.errors.EmptyDataError:
        # User defined start and end dates, ticker, and interval (not implemented yet)

        start_date = input("Enter a start date (YYYY-MM-DD): ")
        end_date = input("Enter an end date (YYYY-MM-DD): ")
        xinterval = input("Enter a time interval (1h, 1d, etc.): ")
        t = input("Enter a ticker: ")

        if start_date == "":
            print('Start date invalid, reverting to default (2025-10-03)')
            start_date = "2023-10-01"
        if end_date == "":
            print('End date invalid, reverting to default (2023-10-03)')
            end_date = "2025-10-01"
        if t == "":
            print('Ticker invalid, reverting to default (NVDA)')
            t = 'NVDA'
        if xinterval == "":
            print('Interval invalid, reverting to default (1d)')
            xinterval = "1d"

        data = pd.DataFrame({
            'ticker': [t], 
            'start': [start_date], 
            'end': [end_date],
            'xinterval': [xinterval]
            })
        data.to_csv('x_utilities/utils.csv', index=False)
    else:
        t = filedata['ticker'][0]
        start_date = filedata['start'][0]
        end_date = filedata['end'][0]
        xinterval = filedata['xinterval'][0]
    finally:
        return [t, start_date, end_date, xinterval]
    
if __name__ == "__main__":
    getInfo()  # only runs when utils.py is run directly