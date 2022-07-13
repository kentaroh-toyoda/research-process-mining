import re
import glob
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

for xes_filename in glob.glob('../datasets/**/*.xes', recursive=True):
    # set csv filename
    csv_filename = re.sub(r'\.xes$', '.csv', xes_filename)
    print('Converting', xes_filename, 'into', csv_filename)
    # read a xes log file
    log = xes_importer.apply(xes_filename)
    # convert it to a data frame
    dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    print(dataframe.head())
    # save it as csv
    dataframe.to_csv(csv_filename, index=False)
