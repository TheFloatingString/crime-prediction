import json

with open("static/limitespdq.geojson") as f:
	data = json.load(f)

json_output_dict = {}

for index_pdq in range(len(data["features"])):
	pdq_number = int(data["features"][index_pdq]["properties"]["No_PDQ"])
	coordinates = data["features"][index_pdq]["geometry"]["coordinates"][0][0]
	json_output_dict[pdq_number] = coordinates

with open('static/pdq_coords.json', 'w') as outfile:  
    json.dump(json_output_dict, outfile)