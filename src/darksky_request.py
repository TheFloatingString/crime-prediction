import requests

base_url = """https://api.darksky.net/forecast/2dfdcc237ebdb8ba4c7b2360152a9a87/45.508889,-73.561667"""

def farenheit_to_celsius(farenheit):
	return (farenheit-32)*5/9

def get_climate_data(request_url):
	r = requests.get(request_url).json()

	# data = r.content.decode('utf8')
	# print(r)
	# print(data[2])
	# df_head = pd.read_csv(io.StringIO(data), header=0)
	# return df_head

	temp = farenheit_to_celsius(r["currently"]["temperature"])
	wind = r["currently"]["windSpeed"]
	humidity = r["currently"]["humidity"]

	return temp, wind, humidity

# print(get_climate_data(base_url)["currently"])
# print(farenheit_to_celsius(get_climate_data(base_url)["currently"]["temperature"]))
# print(get_climate_data(base_url)["currently"]["windSpeed"])
# print(get_climate_data(base_url)["currently"]["humidity"])
# print(get_climate_data(base_url)["currently"]["humidity"])
