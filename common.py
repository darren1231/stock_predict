# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:47:59 2018

@author: darren
"""

import requests
import pandas as pd
import re


def fetch_common(start_date=[107,2],end_date=[107,3]):
    """
    fetch stock common data from website
    
    Args:
        start_date:
        end_date:
    
    Return:
        data: pandas data fram
    """
    start_year = str(start_date[0] + 1911)
    start_month = str(start_date[1]).zfill(2) 
    
    end_year = str(end_date[0] + 1911)
    end_month = str(end_date[1]).zfill(2)

    requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ':RC4-SHA'
    p = {"ctl00$ContentPlaceHolder1$startText":start_year+"/"+start_month+"/01",
         "ctl00$ContentPlaceHolder1$endText":end_year+"/"+end_month+"/31",
         "ctl00$ContentPlaceHolder1$submitBut": "查詢"}
    r = requests.post("https://www.cnyes.com/twstock/ps_historyprice/0050.htm",p)
    data = pd.read_html(r.content, header=0)[0]
    change_data_format = lambda m: str(int(m.group(0))-1911)    
    data.iloc[:,0]=data.iloc[:,0].str.replace(r'\d+', change_data_format,1)
    
    return data
data = fetch_common()
print (data)
