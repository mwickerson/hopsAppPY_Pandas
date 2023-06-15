"""Hops flask middleware example"""
from flask import Flask
import ghhops_server as hs
import rhino3dm

# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)

# flask app can be used for other stuff drectly
@app.route("/help")
def help():
    return "Welcome to Grashopper Hops for CPython!"

"""
██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ███████╗
██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔════╝
██████╔╝███████║██╔██╗ ██║██║  ██║███████║███████╗
██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║╚════██║
██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║███████║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚══════╝
"""
#NumPy, SciPy, Pandas, Matplotlib, Scikit-Learn, TensorFlow, PyTorch, etc.
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import Scikit-Learn as sk
from pandas import Series, DataFrame
from numpy.random import randn
import math

#tabular and heterogeneous data
#https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html

#Introduction to pandas data structures
#Series and DataFrames are workhorses

"""
███████╗███████╗██████╗ ██╗███████╗███████╗
██╔════╝██╔════╝██╔══██╗██║██╔════╝██╔════╝
███████╗█████╗  ██████╔╝██║█████╗  ███████╗
╚════██║██╔══╝  ██╔══██╗██║██╔══╝  ╚════██║
███████║███████╗██║  ██║██║███████╗███████║
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝
"""

#Series
#Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.).
#example
obj = pd.Series([4, 7, -5, 3])
print(obj)
print(obj.values)
print(obj.index)

#write into a @hops component
@hops.component(
    "/series_05",
    name="Series_04",
    description="Series_04",
    icon="icons/series_04.png",
    inputs=[
        hs.HopsNumber("A", "A", "A"),
        hs.HopsNumber("B", "B", "B"),
        hs.HopsNumber("C", "C", "C"),
        hs.HopsNumber("D", "D", "D"),
    ],
    outputs=[   
        hs.HopsString("Series", "Series", "Series"),
    ],
)
def series_05(A: float, B: float, C: float, D: float):
    obj = pd.Series([A, B, C, D])
    #convert to list
    obj = obj.tolist()
    return obj

#Series from list
#write into a @hops component
@hops.component(
    "/series_13",
    name="Series_07",
    description="Series_07",
    inputs=[
        hs.HopsNumber("list", "list", "list", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Values", "Values", "Values", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],
)
def series_13(list: float):
    obj = pd.Series(list)
    print(type(obj))
    values = obj.values
    print(type(values))
    index = obj.index
    print(type(index))
    obj = obj.tolist()
    values = values.tolist()
    index = index.tolist()
    return obj, values, index
    
#Extension Data Types
#https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#extension-types

#often you want to create a Series  with an index identifying each data point with a label
#example
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)
print(obj2.index)

#write into a @hops component
@hops.component(
    "/series_identifier_01",
    name="Series_Identifier_01",
    description="Series_Identifier_01",
    inputs=[
        hs.HopsNumber("item", "item", "item", access = hs.HopsParamAccess.LIST),
        hs.HopsString("index", "index", "index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Values", "Values", "Values", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],
)
def series_identifier_01(item: float, index: str):
    obj = pd.Series(item, index)
    print(type(obj))
    values = obj.values
    print(type(values))
    index = obj.index
    print(type(index))
    obj = obj.tolist()
    values = values.tolist()
    index = index.tolist()
    return obj, values, index

#compare with NumPy arrays, you can use labels in the index when selecting single values or a set of values
#example
print(obj2['a'])
print(obj2['d'])
print(obj2[['c', 'a', 'd']])
print(obj2[obj2 > 0])
print(obj2 * 2)
print(np.exp(obj2))

#write into a @hops component
@hops.component(
    "/single_value_11",
    name="Single_Value_01",
    description="Single_Value_01",
    inputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def single_value_11(Series: float, Index: str):
    obj = pd.Series(Series, Index)
    print(type(obj))
    obj2 = obj['b']
    print(type(obj2))
    return obj2

#write into a @hops component
@hops.component(
    "/set_of_values_02",
    name="Set_of_Values_01",
    description="Set_of_Values_01",
    inputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def set_of_values_02(Series: float, Index: str):
    obj = pd.Series(Series, Index)
    print(type(obj))
    obj2 = obj[['b', 'a', 'd']]
    print(type(obj2))
    #convert to list
    obj2 = obj2.tolist()
    return obj2

#here ['b', 'a', 'd'] is interpreted as a list of indices, 
# even though it contains strings instead of integers

#Using the indexing operator on a Series behaves differently than NumPy arrays
#such as filtering with a boolean array, scalar multiplication, or applying a math function,
# will preserve the index-value link
#example
print(obj2)
print(obj2[obj2 > 0])
print(obj2 * 2)
print(np.exp(obj2))

#write into a @hops component
@hops.component(
    "/boolean_filter_13",
    name="Boolean_Filter_01",
    description="Boolean_Filter_01",
    inputs=[
        hs.HopsNumber("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("higherThan", "higherThan", "higherThan", access = hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def boolean_filter_13(Series: float, Index: str, higherThan: float):
    obj = pd.Series(Series, Index)
    print(type(obj))
    obj2 = obj[obj > higherThan]
    print(type(obj2))
    #convert to list
    obj2 = obj2.tolist()
    return obj2
    
#exponential function
#write into a @hops component
@hops.component(    
    "/exponential_function_03",
    name="Exponential_Function_01",
    description="Exponential_Function_01",
    inputs=[
        hs.HopsNumber("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],  
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def exponential_function_03(Series: float, Index: str):
    obj = pd.Series(Series, Index)
    print(type(obj))
    obj2 = np.exp(obj)
    print(type(obj2))
    #convert to list
    obj2 = obj2.tolist()
    return obj2

#Series as fixed-length, ordered dict
#A Series is a fixed-size dict in that you can get and set values by index label
#example
print('b' in obj2)
print('e' in obj2)

#write into a @hops component
@hops.component(
    "/fixed_length_04",
    name="Fixed_Length_01",
    description="Fixed_Length_01",
    inputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsBoolean("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def fixed_length_04(Series: float, Index: str):
    obj = pd.Series(Series, Index)
    print(type(obj))
    bool_Out = 'b' in obj
    print(type(bool_Out))
    return bool_Out
    
#should you have data contained in a Python dict, you can create a Series from it by passing the dict
#example
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
print(sdata)
obj3 = pd.Series(sdata)
print(obj3)

#write into a @hops component
@hops.component(
    "/dict_03",
    name="Dict_01",
    description="Dict_01",
    inputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def dict_03(Series: float):
    sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    print(sdata)
    obj3 = pd.Series(sdata)
    print(obj3)
    #convert to list
    obj3 = obj3.tolist()
    return obj3

#convvert back to dict
#example
print(obj3)
states = ['California', 'Ohio', 'Oregon', 'Texas']
print(states)
obj4 = pd.Series(sdata, index=states)
print(obj4)

#override index by passing a list of values
#write into a @hops component
@hops.component(
    "/dict_04",
    name="Dict_02",
    description="Dict_02",
    inputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def dict_04(Series: float, Index: str):
    sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    print(sdata)
    obj3 = pd.Series(sdata)
    print(obj3)
    states = ['California', 'Ohio', 'Oregon', 'Texas']
    print(states)
    obj4 = pd.Series(sdata, index=states)
    print(obj4)
    #convert to list
    obj4 = obj4.tolist()
    return obj4

#missing or NA data, it appears as NaN (not a number) which is considered in pandas to mark missing or NA values
#isna and notna functions in pandas should be used to detect missing data
#example
print(pd.isnull(obj4))
print(pd.notnull(obj4))

#write into a @hops component
@hops.component(
    "/missing_data_01",
    name="Missing_Data_01",
    description="Missing_Data_01",
    inputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def missing_data_01(Series: float):
    sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    print(sdata)
    obj3 = pd.Series(sdata)
    print(obj3)
    states = ['California', 'Ohio', 'Oregon', 'Texas']
    print(states)
    obj4 = pd.Series(sdata, index=states)
    print(obj4)
    bool_Out = pd.isnull(obj4)
    print(bool_Out)
    #convert to list
    bool_Out = bool_Out.tolist()
    return bool_Out

#automatically align differently-indexed data in arithmetic operations
#example
print(obj3)
print(obj4)
print(obj3 + obj4)

#write into a @hops component
@hops.component(
    "/auto_align_01",
    name="Auto_Align_01",
    description="Auto_Align_01",
    inputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def auto_align_01(Series: float, Index: str):
    sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    print(sdata)
    obj3 = pd.Series(sdata)
    print(obj3)
    states = ['California', 'Ohio', 'Oregon', 'Texas']
    print(states)
    obj4 = pd.Series(sdata, index=states)
    print(obj4)
    obj5 = obj3 + obj4
    print(obj5)
    #convert to list
    obj5 = obj5.tolist()
    return obj5

#both the Series object itself and its index have a name attribute, 
# which integrates with other key areas of pandas functionality
#example
obj4.name = 'population'
obj4.index.name = 'state'
print(obj4)

#write into a @hops component
@hops.component(
    "/name_attribute_01",
    name="Name_Attribute_01",
    description="Name_Attribute_01",
    inputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def name_attribute_01(Series: float, Index: str):
    sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    print(sdata)
    obj3 = pd.Series(sdata)
    print(obj3)
    states = ['California', 'Ohio', 'Oregon', 'Texas']
    print(states)
    obj4 = pd.Series(sdata, index=states)
    print(obj4)
    obj4.name = 'population'
    obj4.index.name = 'state'
    print(obj4)
    #convert to list
    obj4 = obj4.tolist()
    return obj4

#A Series’s index can be altered in-place by assignment
#example
print(obj)
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
print(obj)

#write into a @hops component
@hops.component(
    "/index_alteration_01",
    name="Index_Alteration_01",
    description="Index_Alteration_01",
    inputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST),
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Value", "Value", "Value", access = hs.HopsParamAccess.LIST)
    ],
)
def index_alteration_01(Series: float, Index: str):
    obj = pd.Series(Series, Index)
    print(type(obj))
    obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
    print(obj)
    #convert to list
    obj = obj.tolist()
    return obj

"""
██████╗  █████╗ ████████╗ █████╗ ███████╗██████╗  █████╗ ███╗   ███╗███████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝
██║  ██║███████║   ██║   ███████║█████╗  ██████╔╝███████║██╔████╔██║█████╗  
██║  ██║██╔══██║   ██║   ██╔══██║██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  
██████╔╝██║  ██║   ██║   ██║  ██║██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
"""
#dataframes are a tabular, spreadsheet-like data structure containing 
#an ordered collection of columns
#each can be a different value type (numeric, string, boolean, etc.)
#the DataFrame has both a row and column index
#think of it as a dict of Series all sharing the same index
#under the hood, the data is stored as one or more two-dimensional blocks
#rather than as a list, dict, or some other collection of one-dimensional arrays
#the exact details of DataFrame’s internals are outside the scope of this book
#hierarchical indexing (also known as multi-indexing) enables you to have multiple
#(two or more) index levels on an axis
#example
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002, 2003],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
print(data)
frame = pd.DataFrame(data)
print(frame)

#write into a @hops component
@hops.component(
    "/dataframe_02",
    name="DataFrame_01",
    description="DataFrame_01",
    inputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
    ],
)
def dataframe_02(state: str, year: float, pop: float):
    data = {'state': state, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    #convert data frame to array
    frame = frame.values
    print(type(frame))
    #convert array to list
    frame = frame.tolist()
    return frame

#write into a @hops component
@hops.component(
    "/dataframe_03b",
    name="DataFrame_01",
    description="DataFrame_01",
    inputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DF1", "DF1", "DF1", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF2", "DF2", "DF2", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF3", "DF3", "DF3", access = hs.HopsParamAccess.LIST)
    ],
)
def dataframe_03b(state: str, year: float, pop: float):
    data = {'state': state, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    #convert data frame to array
    frame = frame.values
    print(type(frame))
    #convert array to list
    frame = frame.tolist()
    #convert to list
    DF1 = frame[0]
    DF2 = frame[1]
    DF3 = frame[2]
    return DF1, DF2, DF3

#supported in a more browser-friendly format in the Jupyter notebook

#head method selects only the first five rows
#write into a @hops component
@hops.component(
    "/head_method_06",
    name="Head_Method_01",
    description="Head_Method_01",
    inputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
    ],
)
def head_method_06(state: str, year: float, pop: float):
    data = {'state': state, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame2 = frame.head()
    print(frame2)
    #convert data frame to array
    frame2 = frame2.values
    print(type(frame2))
    #convert array to list
    frame2 = frame2.tolist()
    return frame2
    
#head method selects only the first five rows
#write into a @hops component
@hops.component(
    "/head_method_07",
    name="Head_Method_01",
    description="Head_Method_01",
    inputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DF1", "DF1", "DF1", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF2", "DF2", "DF2", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF3", "DF3", "DF3", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF4", "DF4", "DF4", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF5", "DF5", "DF5", access = hs.HopsParamAccess.LIST)
    ],
)
def head_method_07(state: str, year: float, pop: float):
    data = {'state': state, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame2 = frame.head()
    print(frame2)
    #convert data frame to array
    frame2 = frame2.values
    print(type(frame2))
    #convert array to list
    frame2 = frame2.tolist()
    #convert to list
    DF1 = frame2[0]
    DF2 = frame2[1]
    DF3 = frame2[2]
    DF4 = frame2[3]
    DF5 = frame2[4]
    return DF1, DF2, DF3, DF4, DF5

#tail method selects only the last five rows
#write into a @hops component
@hops.component(
    "/tail_method_01",
    name="Tail_Method_01",
    description="Tail_Method_01",
    inputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DF1", "DF1", "DF1", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF2", "DF2", "DF2", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF3", "DF3", "DF3", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF4", "DF4", "DF4", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF5", "DF5", "DF5", access = hs.HopsParamAccess.LIST)
    ],
)
def tail_method_01(state: str, year: float, pop: float):
    data = {'state': state, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame2 = frame.tail()
    print(frame2)
    #convert data frame to array
    frame2 = frame2.values
    print(type(frame2))
    #convert array to list
    frame2 = frame2.tolist()
    #convert to list
    DF1 = frame2[0]
    DF2 = frame2[1]
    DF3 = frame2[2]
    DF4 = frame2[3]
    DF5 = frame2[4]
    return DF1, DF2, DF3, DF4, DF5

#sequence of columns can be arranged
#write into a @hops component
@hops.component(
    "/sequence_of_columns_01",
    name="Sequence_of_Columns_01",
    description="Sequence_of_Columns_01",
    inputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DF1", "DF1", "DF1", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF2", "DF2", "DF2", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF3", "DF3", "DF3", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF4", "DF4", "DF4", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF5", "DF5", "DF5", access = hs.HopsParamAccess.LIST)
    ],
)
def sequence_of_columns_01(state: str, year: float, pop: float):
    data = {'state': state, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop'])
    print(frame2)
    #convert data frame to array
    frame2 = frame2.values
    print(type(frame2))
    #convert array to list
    frame2 = frame2.tolist()
    #convert to list
    DF1 = frame2[0]
    DF2 = frame2[1]
    DF3 = frame2[2]
    DF4 = frame2[3]
    DF5 = frame2[4]
    return DF1, DF2, DF3, DF4, DF5  

#pass a column that isn’t contained in data, 
#it will appear with NA values in the result
#write into a @hops component
@hops.component(
    "/column_not_contained_01",
    name="Column_Not_Contained_01",
    description="Column_Not_Contained_01",
    inputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DF1", "DF1", "DF1", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF2", "DF2", "DF2", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF3", "DF3", "DF3", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF4", "DF4", "DF4", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF5", "DF5", "DF5", access = hs.HopsParamAccess.LIST)
    ],
)
def column_not_contained_01(state: str, year: float, pop: float):
    data = {'state': state, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'])
    print(frame2)
    #convert data frame to array
    frame2 = frame2.values
    print(type(frame2))
    #convert array to list
    frame2 = frame2.tolist()
    #convert to list
    DF1 = frame2[0]
    DF2 = frame2[1]
    DF3 = frame2[2]
    DF4 = frame2[3]
    DF5 = frame2[4]
    return DF1, DF2, DF3, DF4, DF5

#dictionary-type notation or by using dot attribute notation

#this is not possible with methods like frame.pop, which will return a Series
#whitespaces in column names won’t work with attribute access

#write into a @hops component
@hops.component(
    "/dictionary_type_notation_03",
    name="Dictionary_Type_Notation_01",
    description="Dictionary_Type_Notation_01",
    inputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST)
    ],
)
def dictionary_type_notation_03(state: str, year: float, pop: float):
    data = {'state': state, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    print(frame["state"])
    s = frame["state"]
    print(type(s))
    #convert to list
    s = s.tolist()
    return s

#dot attribute notation
#write into a @hops component
@hops.component(
    "/dot_attribute_notation_01",
    name="Dot_Attribute_Notation_01",
    description="Dot_Attribute_Notation_01",
    inputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("state", "state", "state", access = hs.HopsParamAccess.LIST)
    ],
)
def dot_attribute_notation_01(state: str, year: float, pop: float):
    data = {'state': state, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    print(frame.state)
    s = frame.state
    print(type(s))
    #convert to list
    s = s.tolist()
    return s

#iloc method enables you to select a subset of the rows and columns from a DataFrame
#write into a @hops component
@hops.component(
    "/iloc_method_02",
    name="Iloc_Method_01",
    description="Iloc_Method_01",
    inputs=[
        hs.HopsString("stateNum", "stateNum", "stateNum", access = hs.HopsParamAccess.LIST),
        hs.HopsString("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsString("pop", "pop", "pop", access = hs.HopsParamAccess.LIST),
        hs.HopsInteger("index", "index", "index", access = hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
    ],
)
def iloc_method_02(stateNum: str, year: str, pop: str, index: int):
    data = {'stateNum': stateNum, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame2 = frame.iloc[index]
    print(frame2)
    #convert data frame to array
    frame2 = frame2.values
    print(type(frame2))
    #convert array to list
    frame2 = frame2.tolist()
    return frame2

#modify columns by assignment
#write into a @hops component
@hops.component(
    "/modify_columns_04",
    name="Modify_Columns_01",
    description="Modify_Columns_01",
    inputs=[
        hs.HopsString("stateNum", "stateNum", "stateNum", access = hs.HopsParamAccess.LIST),
        hs.HopsString("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsString("pop", "pop", "pop", access = hs.HopsParamAccess.LIST),
        hs.HopsInteger("index", "index", "index", access = hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
    ],
)
def modify_columns_04(stateNum: str, year: str, pop: str, index: int):
    data = {'stateNum': stateNum, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame['debt'] = 16.5
    print(frame)
    frame['debt'] = np.arange(index)
    print(frame)
    #convert data frame to array
    frame = frame.values
    print(type(frame))
    #convert array to list
    frame = frame.tolist()
    return frame

#write into a @hops component
@hops.component(
    "/modify_columns_05",
    name="Modify_Columns_01",
    description="Modify_Columns_01",
    inputs=[
        hs.HopsString("stateNum", "stateNum", "stateNum", access = hs.HopsParamAccess.LIST),
        hs.HopsString("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsString("pop", "pop", "pop", access = hs.HopsParamAccess.LIST),
        hs.HopsInteger("index", "index", "index", access = hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("DF1", "DF1", "DF1", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF2", "DF2", "DF2", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF3", "DF3", "DF3", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF4", "DF4", "DF4", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF5", "DF5", "DF5", access = hs.HopsParamAccess.LIST)
    ],
)
def modify_columns_04(stateNum: str, year: str, pop: str, index: int):
    data = {'stateNum': stateNum, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame['debt'] = 16.5
    print(frame)
    frame['debt'] = np.arange(index)
    print(frame)
    #convert data frame to array
    frame = frame.values
    print(type(frame))
    #convert array to list
    frame = frame.tolist()
    #convert to list
    DF1 = frame[0]
    DF2 = frame[1]
    DF3 = frame[2]
    DF4 = frame[3]
    DF5 = frame[4]
    return DF1, DF2, DF3, DF4, DF5

#value length must match DataFrame length
#write into a @hops component
@hops.component(
    "/modify_columns_NAN_01",
    name="Modify_Columns_NAN_01",
    description="Modify_Columns_NAN_01",
    inputs=[
        hs.HopsString("stateNum", "stateNum", "stateNum", access = hs.HopsParamAccess.LIST),
        hs.HopsString("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsString("pop", "pop", "pop", access = hs.HopsParamAccess.LIST),
        hs.HopsInteger("index", "index", "index", access = hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
    ],
)
def modify_columns_NAN_01(stateNum: str, year: str, pop: str, index: int):
    data = {'stateNum': stateNum, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    val = pd.Series([-1.2, -1.5, -1.7], index=[0, 2, 4])
    print(val)
    frame['debt'] = val
    print(frame)
    #convert data frame to array
    frame = frame.values
    print(type(frame))
    #convert array to list
    frame = frame.tolist()
    return frame

#assigning a column that doesn’t exist will create a new column
#del method will delete columns as with a dict
#new columns cannot be added with the dot notation
#write into a @hops component
@hops.component(
    "/del_method_03",
    name="Del_Method_01",
    description="Del_Method_01",
    inputs=[
        hs.HopsString("stateNum", "stateNum", "stateNum", access = hs.HopsParamAccess.LIST),
        hs.HopsString("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsString("pop", "pop", "pop", access = hs.HopsParamAccess.LIST),
        hs.HopsInteger("index", "index", "index", access = hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
    ],
)
def del_method_03(stateNum: str, year: str, pop: str, index: int):
    data = {'stateNum': stateNum, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame['eastern'] = frame.stateNum == '2'
    print(frame)
    del frame['eastern']
    print(frame)
    #convert data frame to array
    frame = frame.values
    print(type(frame))
    #convert array to list
    frame = frame.tolist()
    return frame

#explicitly copied with the Series’s copy method
#write into a @hops component
@hops.component(
    "/copy_method_01",
    name="Copy_Method_01",
    description="Copy_Method_01",
    inputs=[
        hs.HopsString("stateNum", "stateNum", "stateNum", access = hs.HopsParamAccess.LIST),
        hs.HopsString("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsString("pop", "pop", "pop", access = hs.HopsParamAccess.LIST),
        hs.HopsInteger("index", "index", "index", access = hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
    ],
)
def copy_method_01(stateNum: str, year: str, pop: str, index: int):
    data = {'stateNum': stateNum, 'year': year, 'pop': pop}
    print(data)
    frame = pd.DataFrame(data)
    print(frame)
    frame2 = frame.copy()
    print(frame2)
    frame2['debt'] = 16.5
    print(frame2)
    print(frame)
    #convert data frame to array
    frame2 = frame2.values
    print(type(frame2))
    #convert array to list
    frame2 = frame2.tolist()
    return frame2

#nested dicts of dicts
#write into a @hops component
@hops.component(
    "/nested_dicts_01",
    name="Nested_Dicts_01",
    description="Nested_Dicts_01",
    inputs=[
        hs.HopsString("stateNum", "stateNum", "stateNum", access = hs.HopsParamAccess.LIST),
        hs.HopsString("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsString("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
    ],
)
def nested_dicts_01(stateNum: str, year: str, pop: str):
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    print(pop)
    frame3 = pd.DataFrame(pop)
    print(frame3)
    #convert data frame to array
    frame3 = frame3.values
    print(type(frame3))
    #convert array to list
    frame3 = frame3.tolist()
    return frame3

#nested dicts of dicts
#write into a @hops component
@hops.component(
    "/nested_dicts_02",
    name="Nested_Dicts_01",
    description="Nested_Dicts_01",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("DF1", "DF1", "DF1", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF2", "DF2", "DF2", access = hs.HopsParamAccess.LIST),
        hs.HopsString("DF3", "DF3", "DF3", access = hs.HopsParamAccess.LIST)
        ],
)
def nested_dicts_01(toggle: bool):
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    print(pop)
    frame3 = pd.DataFrame(pop)
    print(frame3)
    #convert data frame to array
    frame3 = frame3.values
    print(type(frame3))
    #convert array to list
    frame3 = frame3.tolist()
    #convert to list
    DF1 = frame3[0]
    DF2 = frame3[1]
    DF3 = frame3[2]
    return DF1, DF2, DF3

#transpose the DataFrame (swap rows and columns)
#you could loose data if a column contains mixed types and you transpose it back and forth
#write into a @hops component
@hops.component(
    "/transpose_01",
    name="Transpose_01",
    description="Transpose_01",
    inputs=[
        hs.HopsString("stateNum", "stateNum", "stateNum", access = hs.HopsParamAccess.LIST),
        hs.HopsString("year", "year", "year", access = hs.HopsParamAccess.LIST),
        hs.HopsString("pop", "pop", "pop", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
    ],
)
def transpose_01(stateNum: str, year: str, pop: str):
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    print(pop)
    frame3 = pd.DataFrame(pop)
    print(frame3)
    frame3 = frame3.T
    print(frame3)
    #convert data frame to array
    frame3 = frame3.values
    print(type(frame3))
    #convert array to list
    frame3 = frame3.tolist()
    return frame3

#keys in the inner dicts are combined and sorted to form the index in the result
#this isn't true if an explicit index is specified
#write into a @hops component
@hops.component(
    "/keys_in_inner_dicts_01",
    name="Keys_In_Inner_Dicts_01",
    description="Keys_In_Inner_Dicts_01",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
        ],
)
def keys_in_inner_dicts_01(toggle: bool):
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    print(pop)
    frame3 = pd.DataFrame(pop, index=[2001, 2002, 2003])
    print(frame3)
    #convert data frame to array
    frame3 = frame3.values
    print(type(frame3))
    #convert array to list
    frame3 = frame3.tolist()
    return frame3

#dicts of Series
#write into a @hops component
@hops.component(
    "/dicts_of_series_01",
    name="Dicts_Of_Series_01",
    description="Dicts_Of_Series_01",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
        ],
)
def dicts_of_series_01(toggle: bool):
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    print(pop)
    frame3 = pd.DataFrame(pop, index=[2001, 2002, 2003])
    pop = {'Nevada': frame3['Nevada'][:-1],
    'Ohio': frame3['Ohio'][:2]}
    print(pop)
    frame3 = pd.DataFrame(pop)
    print(frame3)
    #convert data frame to array
    frame3 = frame3.values
    print(type(frame3))
    #convert array to list
    frame3 = frame3.tolist()
    return frame3

#Possible data inputs to the DataFrame constructor
#2D ndarray - A matrix of data, passing optional row and column labels
#dict of arrays, lists, or tuples - Each sequence becomes a column in the DataFrame. All sequences must be the same length
#NumPy structured/record array - Treated as the “dict of arrays” case
#dict of Series - Each value becomes a column. Indexes from each Series are unioned together to form the result’s row index if no explicit index is passed
#dict of dicts - Each inner dict becomes a column. Keys are unioned to form the row index as in the “dict of Series” case
#list of dicts or Series - Each item becomes a row in the DataFrame. Union of dict keys or Series indexes become the DataFrame’s column labels
#List of lists or tuples - Treated as the “2D ndarray” case
#Another DataFrame - The DataFrame’s indexes are used unless different ones are passed
#NumPy MaskedArray - Like the “2D ndarray” case except masked values become NA/missing in the DataFrame result

#if a DataFrame’s index and columns have their name attributes set, 
# these will also be displayed
#write into a @hops component
@hops.component(
    "/data_inputs_01",
    name="Data_Inputs_01",
    description="Data_Inputs_01",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
        ],
)
def data_inputs_01(toggle: bool):
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    print(pop)
    frame3 = pd.DataFrame(pop, index=[2001, 2002, 2003])
    print(frame3)
    frame3.index.name = 'year'
    frame3.columns.name = 'state'
    print(frame3)
    #convert data frame to array
    frame3 = frame3.values
    print(type(frame3))
    #convert array to list
    frame3 = frame3.tolist()
    return frame3

#unlike Series, DataFrame does not have a name attribute
#DataFrame's to_numpy method returns the data contained in the 
# DataFrame as a two-dimensional ndarray
#write into a @hops component
@hops.component(
    "/to_numpy_01",
    name="To_Numpy_01",
    description="To_Numpy_01",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
        ],
)
def to_numpy_01(toggle: bool):
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    print(pop)
    frame3 = pd.DataFrame(pop, index=[2001, 2002, 2003])
    print(frame3)
    print(frame3.to_numpy())
    #convert data frame to array
    frame3 = frame3.values
    print(type(frame3))
    #convert array to list
    frame3 = frame3.tolist()
    return frame3

"""
██╗███╗   ██╗██████╗ ███████╗██╗  ██╗                     
██║████╗  ██║██╔══██╗██╔════╝╚██╗██╔╝                     
██║██╔██╗ ██║██║  ██║█████╗   ╚███╔╝                      
██║██║╚██╗██║██║  ██║██╔══╝   ██╔██╗                      
██║██║ ╚████║██████╔╝███████╗██╔╝ ██╗                     
╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝                     
                                                          
 ██████╗ ██████╗      ██╗███████╗ ██████╗████████╗███████╗
██╔═══██╗██╔══██╗     ██║██╔════╝██╔════╝╚══██╔══╝██╔════╝
██║   ██║██████╔╝     ██║█████╗  ██║        ██║   ███████╗
██║   ██║██╔══██╗██   ██║██╔══╝  ██║        ██║   ╚════██║
╚██████╔╝██████╔╝╚█████╔╝███████╗╚██████╗   ██║   ███████║
 ╚═════╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   ╚══════╝
"""   
#pandas's index objects are responsible for holding 
# the axis labels and other metadata 
# (including the DataFrame’s index (row labels) and columns) and 
# other metadata (like the axis name or names)
#any other sequence of labels used when constructing a Series or DataFrame 
#is internally converted to an Index
#write into a @hops component
@hops.component(
    "/index_objects_01",
    name="Index_Objects_01",
    description="Index_Objects_01",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST)
        ],
)
def index_objects_01(toggle: bool):
    obj = pd.Series(range(3), index=['a', 'b', 'c'])
    print(obj)
    index = obj.index
    print(index)
    print(index[1:])
    #convert data frame to array
    index = index.values
    print(type(index))
    #convert array to list
    index = index.tolist()
    return index[1:]

#immutable, cannot be modified by the user
#immutability makes it safer to share Index objects among data structures
#write into a @hops component
@hops.component(
    "/index_objects_02",
    name="Index_Objects_02",
    description="Index_Objects_02",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("Series", "Series", "Series", access = hs.HopsParamAccess.LIST)
        ],
)
def index_objects_02(toggle: bool):
    obj = pd.Series(range(3), index=['a', 'b', 'c'])
    print(obj)
    index = obj.index
    print(index)
    #index[1] = 'd' #TypeError: Index does not support mutable operations
    #print(index)
    #convert data frame to array
    index = index.values
    print(type(index))
    #convert array to list
    index = index.tolist()
    return index

#take advantage of pandas’s capabilities provided by Index objects
#it is important to know that the Index objects are immutable

#Index also behaves like a fixed-size set
#write into a @hops component
@hops.component(
    "/index_objects_03",
    name="Index_Objects_03",
    description="Index_Objects_03",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("Ohio_Output", "Ohio_Output", "Ohio_Output", access = hs.HopsParamAccess.LIST),
        hs.HopsString("2003_Output", "2003_Output", "2003_Output", access = hs.HopsParamAccess.LIST)
        ],
)
def index_objects_03(toggle: bool):
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    print(pop)
    frame3 = pd.DataFrame(pop, index=[2001, 2002, 2003])
    print(frame3.columns)
    Ohio_Output = ('Ohio' in frame3.columns)
    print(Ohio_Output)
    i2003_Output = (2003 in frame3.index)
    print(i2003_Output)
    #convert data frame to array
    index = frame3.index.values
    print(type(index))
    #convert array to list
    index = index.tolist()
    return Ohio_Output, i2003_Output

#Unlike Python sets, a pandas Index can contain duplicate labels
#write into a @hops component
@hops.component(
    "/index_objects_08",
    name="Index_Objects_04",
    description="Index_Objects_04",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("DataFrame", "DataFrame", "DataFrame", access = hs.HopsParamAccess.LIST)
        ],
)
def index_objects_08(toggle: bool):
    frame3 = pd.Series(range(5), index=['foo', 'foo', 'bar', 'bar', 'baz'])
    #convert data frame to array
    index = frame3.index.values
    print(type(index))
    #convert array to list
    index = index.tolist()
    return index

#Each Index has a number of methods and properties for set logic 
# and answering other common questions about the data it contains

#Some useful Index methods and properties are summarized in the following table
#append - Concatenate with additional Index objects, producing a new Index
#difference - Compute set difference as an Index
#intersection - Compute set intersection
#union - Compute set union
#isin - Compute boolean array indicating whether each value is contained in the passed collection
#delete - Compute new Index with element at index i deleted
#drop - Compute new index by deleting passed values
#insert - Compute new Index by inserting element at index i
#is_monotonic - Returns True if each element is greater than or equal to the previous element
#is_unique - Returns True if the Index has no duplicate values
#unique - Compute the array of unique values in the Index

#append - Concatenate with additional Index objects, producing a new Index
#write into a @hops component
@hops.component(
    "/append_01",
    name="Append_01",
    description="Append_01",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
        hs.HopsString("index", "index", "index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
        ],
)
def append_01(toggle: bool, index: str):
    index = pd.Index(['c', 'a', 'b'])
    print(index)
    index2 = pd.Index(['d', 'e', 'f'])
    print(index2)
    index3 = index.append(index2)
    print(index3)
    #convert data frame to array
    index3 = index3.values
    print(type(index3))
    #convert array to list
    index3 = index3.tolist()
    return index3

#append doesn’t modify the original Indexes!
#different - compute set difference as an Index
#write into a @hops component
@hops.component(
    "/difference_01",
    name="Difference_01",
    description="Difference_01",
    inputs=[
        hs.HopsBoolean("toggle", "toggle", "toggle", access = hs.HopsParamAccess.LIST),
        hs.HopsString("index", "index", "index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Index", "Index", "Index", access = hs.HopsParamAccess.LIST)
        ],
)
def difference_01(toggle: bool, index: str):
    index = pd.Index(['c', 'a', 'b'])
    print(index)
    index2 = pd.Index(['c', 'a', 'e'])
    print(index2)
    index3 = index.difference(index2)
    print(index3)
    #convert data frame to array
    index3 = index3.values
    print(type(index3))
    #convert array to list
    index3 = index3.tolist()
    return index3

#work these out for yourself...

#intersection - Compute set intersection
#union - Compute set union
#isin - Compute boolean array indicating whether each value is contained in the passed collection
#delete - Compute new Index with element at index i deleted
#drop - Compute new index by deleting passed values
#insert - Compute new Index by inserting element at index i
#is_monotonic - Returns True if each element is greater than or equal to the previous element
#is_unique - Returns True if the Index has no duplicate values
#unique - Compute the array of unique values in the Index

"""
███████╗███████╗███████╗███████╗███╗   ██╗████████╗██╗ █████╗ ██╗                                      
██╔════╝██╔════╝██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║██╔══██╗██║                                      
█████╗  ███████╗███████╗█████╗  ██╔██╗ ██║   ██║   ██║███████║██║                                      
██╔══╝  ╚════██║╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██╔══██║██║                                      
███████╗███████║███████║███████╗██║ ╚████║   ██║   ██║██║  ██║███████╗                                 
╚══════╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═╝╚══════╝                                 
                                                                                                       
███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗     ██╗████████╗██╗   ██╗
██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔══██╗██║     ██║╚══██╔══╝╚██╗ ██╔╝
█████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████║██║     ██║   ██║    ╚████╔╝ 
██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║██╔══██║██║     ██║   ██║     ╚██╔╝  
██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║██║  ██║███████╗██║   ██║      ██║   
╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝ 
"""
#Essential functionality
#interacting with the data contained in Series and DataFrame objects
#data analysis and manipulation using pandas
#focus on data loading, preparation, and joining/merging

#Reindexing
#pandas objects are equipped with a reindex method that
# rearranges the data according to the new index
#introduces missing values if any index values were not already present
#write into a @hops component
@hops.component(
    "/reindexing_02",
    name="Reindexing_01",
    description="Reindexing_01",
    inputs=[
        hs.HopsString("list1", "list1", "list1", access = hs.HopsParamAccess.LIST),
        hs.HopsString("index", "index", "index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("DF1", "DF1", "DF1", access = hs.HopsParamAccess.LIST)
        ],
)
def reindexing_02(list1: str, index: str):
    obj = pd.Series(list1, index=index)
    print(obj)
    obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
    print(obj2)
    #convert data frame to array
    obj2 = obj2.values  
    print(type(obj2))
    #convert array to list
    obj2 = obj2.tolist()
    return obj2

#ffill method forward-fills the values
#write into a @hops component
@hops.component(
    "/reindexing_ffill_03",
    name="Reindexing_ffill_01",
    description="Reindexing_ffill_01",
    inputs=[
        hs.HopsNumber("list1", "list1", "list1", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("index", "index", "index", access = hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsNumber("DF1", "DF1", "DF1", access = hs.HopsParamAccess.LIST)
        ],
)
def reindexing_ffill_03(list1: str, index: str):
    obj = pd.Series(list1, index=index)
    print(obj)
    obj2 = obj.reindex(np.arange(6), method='ffill')
    print(obj2)
    #convert data frame to array
    obj2 = obj2.values
    print(type(obj2))
    #convert array to list
    obj2 = obj2.tolist()
    return obj2

#bfill method backward-fills the values

#With DataFrame, reindex can alter either the (row) index, columns, or both

"""
███╗   ███╗ █████╗ ███████╗███████╗
████╗ ████║██╔══██╗╚══███╔╝██╔════╝
██╔████╔██║███████║  ███╔╝ █████╗  
██║╚██╔╝██║██╔══██║ ███╔╝  ██╔══╝  
██║ ╚═╝ ██║██║  ██║███████╗███████╗
╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
"""
#write a walker algorithm to build a 3D labyrinth
# df_maze.py
import random

# Create a maze using the depth-first algorithm described at
# https://scipython.com/blog/making-a-maze/
# Christian Hill, April 2017.

class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny, ix=0, iy=0):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """

        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.maze_map[x][y]

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        maze_rows = ['-' * self.nx * 2]
        for y in range(self.ny):
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)

    def write_svg(self, filename):
        """Write an SVG image of the maze to filename."""

        aspect_ratio = self.nx / self.ny
        # Pad the maze all around by this amount.
        padding = 10
        # Height and width of the maze image (excluding padding), in pixels
        height = 500
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx

        def write_wall(ww_f, ww_x1, ww_y1, ww_x2, ww_y2):
            """Write a single wall to the SVG image file handle f."""

            print('<line x1="{}" y1="{}" x2="{}" y2="{}"/>'
                  .format(ww_x1, ww_y1, ww_x2, ww_y2), file=ww_f)

        # Write the SVG image file for maze
        with open(filename, 'w') as f:
            # SVG preamble and styles.
            print('<?xml version="1.0" encoding="utf-8"?>', file=f)
            print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
            print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
            print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
                  .format(width + 2 * padding, height + 2 * padding,
                          -padding, -padding, width + 2 * padding, height + 2 * padding),
                  file=f)
            print('<defs>\n<style type="text/css"><![CDATA[', file=f)
            print('line {', file=f)
            print('    stroke: #000000;\n    stroke-linecap: square;', file=f)
            print('    stroke-width: 5;\n}', file=f)
            print(']]></style>\n</defs>', file=f)
            # Draw the "South" and "East" walls of each cell, if present (these
            # are the "North" and "West" walls of a neighbouring cell in
            # general, of course).
            for x in range(self.nx):
                for y in range(self.ny):
                    if self.cell_at(x, y).walls['S']:
                        x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
                    if self.cell_at(x, y).walls['E']:
                        x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
            # Draw the North and West maze border, which won't have been drawn
            # by the procedure above.
            print('<line x1="0" y1="0" x2="{}" y2="0"/>'.format(width), file=f)
            print('<line x1="0" y1="0" x2="0" y2="{}"/>'.format(height), file=f)
            print('</svg>', file=f)

    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1

#write into a @hops component
@hops.component(
    "/Maze_07",
    name="Maze_01",
    description="Maze_01",
    inputs=[
        hs.HopsInteger("nx", "nx", "nx", access = hs.HopsParamAccess.ITEM),
        hs.HopsInteger("ny", "ny", "ny", access = hs.HopsParamAccess.ITEM),
        hs.HopsInteger("ix", "ix", "ix", access = hs.HopsParamAccess.ITEM),
        hs.HopsInteger("iy", "iy", "iy", access = hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("Maze", "Maze", "Maze", access = hs.HopsParamAccess.ITEM)
    ],
)
def Maze_07(nx: int, ny: int, ix: int, iy: int):
    # Create a maze and write it to a SVG file.
    maze = Maze(nx, ny, ix, iy)
    maze.make_maze()
    maze.write_svg('maze.svg')
    return ("Congradulations, you just made a maze!")

#Bagels, by Al Sweigart al@inventwithpython
#A deductive logic game where you must guess a number based on clues
#View this code at https://nostarch.com/big-book-small-python-projects
#A version of this game is featured in the book "Invent Your Own
#Computer Games with Python" https://nostarch.com/inventwithpython
#Tags - short, game, puzzle

import random

NUM_DIGITS = 3 #(!) Try setting this to 1 or 10.
MAX_GUESSES = 10 #(!) Try setting this to 1 or 100.

def main():
    print('''Bagels, a deductive logic game.
By Al Sweigart al@inventwithpython

I am thinking of a {}-digit number with no repeated digits.
Try to guess what it is. Here are some clues:
When I say:    That means:
  Pico         One digit is correct but in the wrong position.
  Fermi        One digit is correct and in the right position.
  Bagels       No digit is correct.

For example, if the secret number was 248 and your guess was 843, the
clues would be Fermi Pico.'''.format(NUM_DIGITS))

    while True: # Main game loop.
        # This stores the secret number the player needs to guess:
        secretNum = getSecretNum()
        print('I have thought up a number.')
        print('  You have {} guesses to get it.'.format(MAX_GUESSES))

        numGuesses = 1
        while numGuesses <= MAX_GUESSES:
            guess = ''
            # Keep looping until they enter a valid guess:
            while len(guess) != NUM_DIGITS or not guess.isdecimal():
                print('Guess #{}: '.format(numGuesses))
                guess = input('> ')

            clues = getClues(guess, secretNum)
            print(clues)
            numGuesses += 1

            if guess == secretNum:
                # They're correct, so break out of this loop.
                break
            if numGuesses > MAX_GUESSES:
                print('You ran out of guesses.')
                print('The answer was {}.'.format(secretNum))

        # Ask player if they want to play again.
        print('Do you want to play again? (yes or no)')
        if not input('> ').lower().startswith('y'):
            break
    print('Thanks for playing!')

def getSecretNum():
    """Returns a string made up of NUM_DIGITS unique random digits."""
    numbers = list('0123456789') # Create a list of digits 0 to 9.
    random.shuffle(numbers) # Shuffle them into random order.

    # Get the first NUM_DIGITS digits in the list for the secret number:
    secretNum = ''
    for i in range(NUM_DIGITS):
        secretNum += str(numbers[i])
    return secretNum

def getClues(guess, secretNum):
    """Returns a string with the pico, fermi, bagels clues for a guess
    and secret number pair."""
    if guess == secretNum:
        return 'You got it!'

    clues = []

    for i in range(len(guess)):
        if guess[i] == secretNum[i]:
            # A correct digit is in the correct place.
            clues.append('Fermi')
        elif guess[i] in secretNum:
            # A correct digit is in the incorrect place.
            clues.append('Pico')
    if len(clues) == 0:
        return 'Bagels' # There are no correct digits at all.
    else:
        # Sort the clues into alphabetical order so their original order
        # doesn't give information away.
        clues.sort()
        # Make a single string from the list of string clues.
        return ' '.join(clues)

# If the program is run (instead of imported), run the game:
if __name__ == '__main__':

    main()

#write into a @hops component
@hops.component(
    "/Bagels_09",
    name="Bagels_01",
    description="Bagels_01",
    inputs=[
        hs.HopsInteger("NUM_DIGITS", "NUM_DIGITS", "NUM_DIGITS", access = hs.HopsParamAccess.ITEM),
        hs.HopsInteger("MAX_GUESSES", "MAX_GUESSES", "MAX_GUESSES", access = hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("Have fun in the console guessing the number!", "Have fun in the console guessing the number!", "Have fun in the console guessing the number!", access = hs.HopsParamAccess.ITEM)
    ],
)
def Bagels_09(NUM_DIGITS: int, MAX_GUESSES: int):
    print('''Bagels, a deductive logic game.
By Al Sweigart al@inventwithpython

I am thinking of a {}-digit number with no repeated digits.
Try to guess what it is. Here are some clues:
When I say:    That means:
  Pico         One digit is correct but in the wrong position.
  Fermi        One digit is correct and in the right position.
  Bagels       No digit is correct.

For example, if the secret number was 248 and your guess was 843, the
clues would be Fermi Pico.'''.format(NUM_DIGITS))

    while True: # Main game loop.
        # This stores the secret number the player needs to guess:
        secretNum = getSecretNum()
        print('I have thought up a number.')
        print('  You have {} guesses to get it.'.format(MAX_GUESSES))

        numGuesses = 1
        while numGuesses <= MAX_GUESSES:
            guess = ''
            # Keep looping until they enter a valid guess:
            while len(guess) != NUM_DIGITS or not guess.isdecimal():
                print('Guess #{}: '.format(numGuesses))
                guess = input('> ')

            clues = getClues(guess, secretNum)
            print(clues)
            numGuesses += 1

            if guess == secretNum:
                # They're correct, so break out of this loop.
                break
            if numGuesses > MAX_GUESSES:
                print('You ran out of guesses.')
                print('The answer was {}.'.format(secretNum))

        # Ask player if they want to play again.
        print('Do you want to play again? (yes or no)')
        if not input('> ').lower().startswith('y'):
            break
    return ('Thanks for playing!')


#With DataFrame, reindex can alter either the (row) index, columns, or both
#write into a @hops component






















if __name__ == "__main__":
    app.run(debug=True)