import pandas as pd
import numpy as np

# Q1
print("Q1 : " , pd.__version__)

house_df = pd.read_csv('CaliforniaHousingPrices.csv')
# Q2
print("Q2 : " , house_df.columns.size)

# Q3 total bedrooms have NaN values
print("Q3 : total_bedrooms")
# Q4
print("Q4 : " , len(house_df['ocean_proximity'].unique()))

# Q5
near_bay_df = house_df[house_df['ocean_proximity'] == "NEAR BAY"]
print("Q5 : " , near_bay_df['median_house_value'].mean())

# Q6
before_fillna = house_df['total_bedrooms'].mean()
house_df['total_bedrooms'] = house_df['total_bedrooms'].fillna(before_fillna)
after_fillna = house_df['total_bedrooms'].mean()
print("Q6 : before fillna -> ",before_fillna," , after fillna -> ", after_fillna)

# Q7 
island_only = house_df[house_df['ocean_proximity'] == "ISLAND"]
x = island_only[['housing_median_age','total_rooms','total_bedrooms']]
x_transpose = x.T
xtx = x_transpose.dot(x)

y = np.array([950, 1300, 800, 1000, 1300])

inv_xtx = pd.DataFrame(np.linalg.inv(xtx.values), xtx.columns, xtx.index)

w = inv_xtx.dot(x_transpose)
w= w.dot(y)
print("Q7 : ",w[-1])



