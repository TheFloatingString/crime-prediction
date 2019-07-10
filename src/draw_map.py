"""
Function to parse string that will be used by Google Maps API to draw maps in JS of PDQ
"""

# import modules
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

with open("static/pdq_coords.json") as f:
	data = json.load(f)

with open("static/level_color_dict.json") as f:
	level_color_dict = json.load(f)

def return_coords(X_input_data=None):
	counter = 0
	output_str = ''
	for pdq_identifier in list(data.keys()):
		clf = joblib.load("static/saved_models/v2_rn_forest_"+str(pdq_identifier)+".0.pkl")
		prediction = list(clf.predict([X_input_data])[0])
		print(prediction)
		indices = [i for i, x in enumerate(prediction) if x == max(prediction)]

		predicted_level = indices[-1]
		print(predicted_level)
		hex_color_code = level_color_dict[str(int(predicted_level))]

		output_str += f"""					        var triangleCoords{str(counter)} = ["""
		# print(data[pdq_identifier])
		for row in data[pdq_identifier]:
			output_str += """{lat: """+str(row[1])+""", lng: """+str(row[0])+"""},"""
		output_str = output_str[:-1]
		output_str += """];"""
		output_str += """					        var pdqPolygon"""+str(counter)+""" = new google.maps.Polygon({
					paths: triangleCoords"""+str(counter)+""",
					strokeColor: '#ffffff',
					strokeOpacity: 0.85,
					strokeWeight: 2,
					fillColor: '"""+hex_color_code+"""',
					fillOpacity: 0.55
					});
					pdqPolygon"""
		output_str+=str(counter)+""".setMap(map);"""
		output_str+="google.maps.event.addListener(pdqPolygon"+str(counter)+",\"click\",function() {							$.get(\"/get_pdq_data/"+str(pdq_identifier)+".0\", function(data) {console.log($.parseJSON(data));myBarChart.data.datasets[0].data = $.parseJSON(data);myBarChart.update();document.getElementById(\"p1\").innerHTML = \""+pdq_identifier+"\";document.getElementById(\"pdq_num\").innerHTML = \""+pdq_identifier+"\";document.getElementById(\"redirect_pdq_link\").href=\"https://spvm.qc.ca/en/pdq"+str(int(pdq_identifier))+"\";})});"
		counter += 1
	return output_str

# document.getElementById("abc").href="xyz.php";

def compute_level(X_input_data, pdq_identifier):
	clf = joblib.load("static/saved_models/v2_rn_forest_"+str(pdq_identifier)+".pkl")
	prediction = list(clf.predict([X_input_data])[0])

	return str(prediction)
