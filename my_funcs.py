from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt

#Function to plot 2 variables, either colored by a 3rd varaible or not.
#if save_image is set to True, the plot will be saved as an image, named by
#label1_label2
def plot_2_vars(var1, var2, var_color, data, save_image, label1, label2):
	#Get all features to plot, and discover what's of interest and what's not.
	features_for_testing=[var1, var2]
	if(var_color!='b'):
		features_for_testing.append(var_color)
		x1=[data[x][var1] for x in data if data[x]['poi']==1]
		y1=[data[x][var2] for x in data if data[x]['poi']==1]
		x2=[data[x][var1] for x in data if data[x]['poi']==0]
		y2=[data[x][var2] for x in data if data[x]['poi']==0]
		
		plt.scatter(x1, y1, c='r', label="POI")
		plt.scatter(x2, y2, c='b', label="Non-POI")

	else:
		x1=[data[x][var1] for x in data]
		y1=[data[x][var2] for x in data]
		plt.scatter(x1, y1, c= var_color)
	
	plt.xlabel(label1)
	plt.ylabel(label2)
	plt.legend()
	if(save_image):
		plt.savefig(label1+"_"+label2+".png")
	plt.show()

#function to change NaNs into 0 if no return val is passed, or return val
def repair_nan(value, ret_val=0):
	if(value=='NaN'):
		return ret_val
	else:
		return value
