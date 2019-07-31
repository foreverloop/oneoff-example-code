import requests,json,csv,ast,datetime
import pandas as pd

with open("endpoint-longs-lat.csv","r") as f:
	reader = csv.reader(f,delimiter=',')
	for row in reader:
 		endpoint = row[0]

longlats = json.loads(requests.get(endpoint).text)

data_list = []
for idx,row in enumerate(longlats["Locations"]["Location"]):
	region = row.get('region', 'none')
	auth_area = row.get('unitaryAuthArea','none')
	elevation = row.get('elevation',7777) # dummy replace with mean later
	
	add_col = [row['name'],row['latitude'],row['longitude'],region,auth_area,elevation]
	data_list.append(add_col)


df = pd.DataFrame(data_list,columns=['name','latitude','longitude','region','auth_area','elevation'])
df.to_csv("location_detail_final.csv",mode='a',header=True)