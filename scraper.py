# -*- coding: utf-8 -*-
import pandas as pd
import sys
import requests

pd.options.display.encoding = sys.stdout.encoding

class scraper_stock(object):
    
    def __init__(self):
        
        pass
        
    def fetch_single_table(self,website_url,year,month,stock_code):
        """
        According to website_url, get the information of stock table
        
        Args:
            website_url : ex:'https://stock.wearn.com/netbuy.asp?Year={}&month={}&kind={}'
            year:
            month:
            stock_num
            
        Retrun:
            data: pandas datafram
        """
        url = website_url.format(year,month,stock_code)
        data = pd.read_html(url)[0]
        
        data=data.drop(data.index[0:2])
        data=data.drop_duplicates(data,keep='first')
    
        return data
    
    def fetch_range_period(self,website_url,stock_code,start_date,end_date):
        """
        According to given date range, get the data from website
        
        Args:
            website_url : ex:'https://stock.wearn.com/netbuy.asp?Year={}&month={}&kind={}'
            start_date:a list format ex:[105,3]
            end_date:a list format ex:[106,3]
        
        Return:
            pd_table: pandas datafram
        """
        
        start_year = start_date[0]
        start_month = start_date[1]
        
        end_year = end_date[0]
        end_month = end_date[1]
        
        
        
        data_list=[]
        for year in range(end_year,start_year-1,-1):
            for month in range(12,0,-1):
                if month>end_month and year==end_year:
                    continue
                else:
                    input_year = str(year)
                    input_month = str(month).zfill(2)
                    
                    data = self.fetch_single_table(website_url,input_year,input_month,stock_code)
                    data_list.append(data)
    #                pd_table=pd.concat(pd_table,data)
#                    print (str(year),str(month).zfill(2))
                
                if year==start_year and month==start_month:
                    break
        pd_table = pd.concat(data_list)
        
        
        return pd_table
        
        
    def fatch_common(self,stock_code,start_date,end_date):
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
        r = requests.post("https://www.cnyes.com/twstock/ps_historyprice/{}.htm".format(stock_code),p)
        data = pd.read_html(r.content, header=0)[0]
        
        change_data_format = lambda m: str(int(m.group(0))-1911)    
        data.iloc[:,0]=data.iloc[:,0].str.replace(r'\d+', change_data_format,1)
        
        return data
        
    
    def get_whole_stock_data(self,stock_code,start_date,end_date):
        """
        Combine all stock information together
        Args:
            start_date:
            end_date:
        
        Return:
            data: pandas data fram
            
        """
        
        
        website_url='https://stock.wearn.com/netbuy.asp?Year={}&month={}&kind={}'
        three_buy_table=self.fetch_range_period(website_url,stock_code,start_date,end_date)
        three_buy_table.columns=[u'日期', u"自營商",u"投信",u"外資"]
        three_buy_table.to_csv("stock.csv",encoding="utf_8_sig")
        print (three_buy_table)
        
        
        website_url='https://stock.wearn.com/acredit.asp?year={}&month={}&kind={}'
        people_table=self.fetch_range_period(website_url,stock_code,start_date,end_date)
        people_table.columns=[u'日期', u"資餘",u"資增減",u"券餘",u"券增減",u"使用率",u"券資比",u"資券抵"]
        people_table.to_csv("margin_trading.csv",encoding="utf_8_sig")
        print (people_table)
        
        common_table = self.fatch_common(stock_code,start_date,end_date)
        common_table.columns=[u'日期', u"開盤價",u"最高價",u"最低價",u"收盤價",u"漲跌",u"漲%",u"成交量",u"成交金額",u"本益比"]
        common_table.to_csv("common.csv",encoding="utf_8_sig")
        print (common_table)
        
        temp_table = pd.merge(left=three_buy_table,right=people_table)
        all_table = pd.merge(left=temp_table,right= common_table, on = "日期")
        
        all_table.to_csv("{}.csv".format(stock_code),encoding="utf_8_sig")

stock_code = "0050"
start_date = [107,2]
end_date = [107,3]




scraper_stock().get_whole_stock_data(stock_code,start_date,end_date)
#data = pd.read_csv("stock.csv")
#print (data)